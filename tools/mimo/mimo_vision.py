#!/usr/bin/env python3
"""tools/mimo/mimo_vision.py - MiMo-V2.6-Pro's own vision tower as Maya's image encoder (the strata-vision protocol).

llama.cpp's mtmd has no MiMo-VL encoder (GQA 32/8 heads, a +-64-token band window that alternates between the row and
the column order of the merged grid, a learned bias on the first key of every windowed layer, 4 full-attention
layers), so this runs the checkpoint's own MiMoVisionTransformer (modeling_mimo_v2.py) with its 364 `visual.*`
tensors in PyTorch, and speaks tools/vision/strata_vision.cpp's protocol so serve/server.py's Vision class drives it:

    mimo_vision.py --mmproj <checkpoint dir> --model <ignored> [--gpu] [--threads N] [--max-tokens N]
    prints "READY 6144", then per stdin line
      ENC <image path> <output path>  ->  "OK <n_tokens> <nx> <ny> <ms>"  or  "ERR <message>"
      QUIT
    output: int32 {'SVE1', n_tokens, nx, ny, 6144} then float32 [n_tokens][6144], row i at (x = i % nx, y = i / nx)

The rows replace the prompt's <|image_pad|> tokens one for one; MiMo positions them like any other token (plain 1-D
RoPE - no M-RoPE), which is what the engine's token path does.  Pre-processing is the checkpoint's own
preprocessor_config.json (Qwen2-VL image processor: 16-px patches, 2x2 merge, ImageNet mean/std), with max_pixels
capped by --max-tokens (one image token = 32x32 px).
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time
import types

# The client must remain stdlib-only: import the tensor stack only in the
# direct encoder or in the single-threaded CPU zygote.
def load_runtime():
    global np, torch
    import numpy as np
    import torch


HF = str(pathlib.Path(__file__).resolve().parent / "hf")   # MiMo-V2 modeling/configuration (Xiaomi, Apache-2.0; see hf/NOTICE)
MAGIC = 0x31455653                             # 'SVE1'


def load_tower(ckpt: pathlib.Path, device: str, dtype: torch.dtype):
    sys.path.insert(0, HF)
    from modeling_mimo_v2 import MiMoVisionTransformer   # noqa: E402
    from safetensors import safe_open                     # noqa: E402

    cfg = json.loads((ckpt / "config.json").read_text())
    vcfg = types.SimpleNamespace(**cfg["vision_config"])
    # no random init: every parameter is loaded below (strict), and the modules' reset_parameters drew 739M random
    # numbers for nothing (3.2 of the ~8 s an on-demand GPU start took); buffers (the rotary table) are built as before
    nn = torch.nn
    saved = {c: c.__dict__["reset_parameters"] for c in vars(nn).values()
             if isinstance(c, type) and issubclass(c, nn.Module) and "reset_parameters" in c.__dict__}
    for c in saved:
        c.reset_parameters = lambda self, *a, **k: None
    try:
        tower = MiMoVisionTransformer(vcfg)
    finally:
        for c, f in saved.items():
            c.reset_parameters = f
    wmap = json.loads((ckpt / "model.safetensors.index.json").read_text())["weight_map"]
    names = [k for k in wmap if k.startswith("visual.")]
    state = {}
    for shard in sorted({wmap[k] for k in names}):
        with safe_open(str(ckpt / shard), framework="pt") as f:
            for k in names:
                if wmap[k] == shard:
                    state[k[len("visual."):]] = f.get_tensor(k)
    # the checkpoint's merger has no biases (ln_q, mlp.0, mlp.2: weights only) - it was trained bias-free, while the
    # module declares them; zero is that model exactly.  Everything else must load one for one.
    for k in ("merger.ln_q.bias", "merger.mlp.0.bias", "merger.mlp.2.bias"):
        if k not in state:
            state[k] = torch.zeros_like(tower.state_dict()[k])
    tower.load_state_dict(state, strict=True)
    return tower.to(device=device, dtype=dtype).eval(), cfg


def make_processor(ckpt: pathlib.Path, max_tokens: int):
    from transformers import Qwen2VLImageProcessor        # noqa: E402

    pc = json.loads((ckpt / "preprocessor_config.json").read_text())
    unit = (pc["patch_size"] * pc["merge_size"]) ** 2     # pixels a merged image token covers
    max_pixels = min(pc["max_pixels"], max_tokens * unit) if max_tokens > 0 else pc["max_pixels"]
    return Qwen2VLImageProcessor(min_pixels=pc["min_pixels"], max_pixels=max_pixels, patch_size=pc["patch_size"],
                                 temporal_patch_size=pc["temporal_patch_size"], merge_size=pc["merge_size"],
                                 image_mean=pc["image_mean"], image_std=pc["image_std"]), pc["merge_size"]


def encode(tower, proc, merge: int, device: str, dtype: torch.dtype, img_path: str):
    from PIL import Image                                  # noqa: E402

    img = (Image.open(img_path) if isinstance(img_path, str) else img_path).convert("RGB")
    inp = proc(images=[img], return_tensors="pt")
    pv, thw = inp["pixel_values"], inp["image_grid_thw"]
    with torch.no_grad():
        out = tower(pv.to(device=device, dtype=dtype), thw.to(device))
    t, h, w = (int(v) for v in thw[0])
    nx, ny = w // merge, h // merge
    rows = out.float().cpu().numpy()
    if rows.shape[0] != t * nx * ny:
        raise ValueError(f"{rows.shape[0]} rows for a {t}x{ny}x{nx} grid")
    return rows, nx, ny


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mmproj", required=True, help="the MiMo-V2.6-Pro checkpoint directory (config + safetensors)")
    ap.add_argument("--model", default="", help="ignored (strata-vision's text-model argument)")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=1024, help="image tokens at most (sets max_pixels)")
    ap.add_argument("--no-flash-attn", action="store_true")
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--zygote", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--zygote-stop", action="store_true", help="stop this argument set's zygote after its workers exit")
    ap.add_argument("--zygote-status", action="store_true", help="report zygote readiness without starting it")
    ap.add_argument("--direct", action="store_true", help="disable zygote use and creation (reference/debug)")
    ap.add_argument("--zygote-idle-timeout", type=float, default=1800.0)
    ap.add_argument("--zygote-log", default=str(pathlib.Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "maya-vision-mimo-zygote.log"))
    a = ap.parse_args()
    if not math.isfinite(a.zygote_idle_timeout) or a.zygote_idle_timeout <= 0:
        ap.error("--zygote-idle-timeout must be positive")
    return a


def run_encoder(a, prepared=None):
    load_runtime()
    if a.measure and not a.gpu:
        print("ERR --measure requires --gpu", flush=True)
        return 1
    torch.set_num_threads(max(1, a.threads))
    device = "cuda:0" if a.gpu and torch.cuda.is_available() else "cpu"
    if a.gpu and device == "cpu":
        print("ERR CUDA unavailable", flush=True)
        return 1
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    if a.measure:
        torch.cuda.init()
        free0, total0 = torch.cuda.mem_get_info()
        torch.cuda.reset_peak_memory_stats()
    ckpt = pathlib.Path(a.mmproj)
    if prepared is None:
        tower, cfg = load_tower(ckpt, device, dtype)
        proc, merge = make_processor(ckpt, a.max_tokens)
    else:
        tower, cfg, proc, merge = prepared
        tower = tower.to(device=device, dtype=dtype)
    n_embd = int(cfg["vision_config"]["out_hidden_size"])
    if a.measure:
        try:
            from PIL import Image
            if a.max_tokens <= 0:
                raise ValueError("--measure requires a positive --max-tokens")
            h = math.isqrt(a.max_tokens)
            while a.max_tokens % h:
                h -= 1
            w = a.max_tokens // h
            for width, height in ((32 * w, 32 * h), (32 * h, 32 * w)):
                rows, _, _ = encode(tower, proc, merge, device, dtype,
                                    Image.new("RGB", (width, height), (128, 128, 128)))
                if len(rows) != a.max_tokens:
                    raise ValueError("warm-up did not reach the image token cap")
            torch.cuda.synchronize()
            free1, _ = torch.cuda.mem_get_info()
            peak = max(torch.cuda.max_memory_reserved(), free0 - free1)
            print(f"MEM {math.ceil(peak * 1.2)} {total0 - free0} {total0}", flush=True)
            return 0
        except Exception as e:
            print("ERR measurement failed: " + str(e).replace("\n", " ")[:300], flush=True)
            return 1
    print(f"READY {n_embd}", flush=True)
    for line in sys.stdin:
        line = line.rstrip("\n")
        if line == "QUIT":
            break
        if not line.startswith("ENC "):
            print("ERR expected ENC <image> <out> or QUIT", flush=True)
            continue
        rest = line[4:]
        sp = rest.rfind(" ")
        img, out = rest[:sp], rest[sp + 1:]
        try:
            t0 = time.time()
            rows, nx, ny = encode(tower, proc, merge, device, dtype, img)
            with open(out, "wb") as f:
                f.write(np.array([MAGIC, rows.shape[0], nx, ny, n_embd], dtype=np.int32).tobytes())
                f.write(np.ascontiguousarray(rows, dtype=np.float32).tobytes())
            print(f"OK {rows.shape[0]} {nx} {ny} {(time.time() - t0) * 1000:.0f}", flush=True)
        except Exception as e:                             # noqa: BLE001 - one bad image must not end the encoder
            print("ERR " + str(e).replace("\n", " ")[:300], flush=True)
    return 0


# Everything below is stdlib-only until run_encoder()/zygote_main().
import ctypes
import fcntl
import hashlib
import os
import select
import selectors
import signal
import socket
import struct
import subprocess

SCRIPT = pathlib.Path(__file__).resolve()


def identity(a):
    return dict(script=str(SCRIPT), mmproj=str(pathlib.Path(a.mmproj).resolve()),
                model=a.model, gpu=a.gpu, threads=a.threads, max_tokens=a.max_tokens,
                no_flash_attn=a.no_flash_attn, no_warmup=a.no_warmup,
                idle_timeout=a.zygote_idle_timeout, log=str(pathlib.Path(a.zygote_log).absolute()))


def socket_paths(a):
    runtime = pathlib.Path(os.environ.get('XDG_RUNTIME_DIR', f'/tmp/maya-vision-mimo-{os.getuid()}'))
    if not runtime.exists():
        runtime.mkdir(mode=0o700, parents=True)
    if runtime.stat().st_uid != os.getuid() or runtime.stat().st_mode & 0o022:
        raise ValueError('runtime directory must be owned by this uid and not writable by others')
    digest = hashlib.sha256(json.dumps(identity(a), sort_keys=True).encode()).hexdigest()[:20]
    path = runtime / f'maya-vision-mimo-{os.getuid()}-{digest}.sock'
    return str(path), str(path.with_suffix('.lock'))


def send_message(conn, obj):
    data = json.dumps(obj).encode()
    conn.sendall(struct.pack('!I', len(data)) + data)


def recv_exact(conn, size):
    result = bytearray()
    while len(result) < size:
        chunk = conn.recv(size - len(result))
        if not chunk:
            raise EOFError('zygote control socket closed')
        result.extend(chunk)
    return bytes(result)


def recv_message(conn):
    length, = struct.unpack('!I', recv_exact(conn, 4))
    if length > 65536:
        raise ValueError('oversized zygote message')
    return json.loads(recv_exact(conn, length))


def connect(a):
    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    conn.settimeout(3)
    try:
        conn.connect(socket_paths(a)[0])
        return conn
    except (FileNotFoundError, ConnectionRefusedError):
        conn.close()
        return None
    except BaseException:
        conn.close()
        raise


def request(conn, op, a, fds=()):
    socket.send_fds(conn, [b'V'], list(fds))
    send_message(conn, dict(op=op, identity=identity(a)))


def daemon_args(a):
    args = [sys.executable, str(SCRIPT), '--zygote', '--mmproj', a.mmproj,
            '--model', a.model, '--threads', str(a.threads), '--max-tokens', str(a.max_tokens),
            '--zygote-idle-timeout', str(a.zygote_idle_timeout), '--zygote-log', a.zygote_log]
    for attr in ('gpu', 'no_flash_attn', 'no_warmup'):
        if getattr(a, attr):
            args.append('--' + attr.replace('_', '-'))
    return args


def spawn_zygote(a):
    # A contender takes flock before heavy imports. Two cold clients may spawn
    # two contenders, but only one imports/loads and owns this socket.
    try:
        with open(a.zygote_log, 'ab', buffering=0) as log:
            subprocess.Popen(daemon_args(a), stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                             start_new_session=True, close_fds=True)
    except OSError as exc:
        print(f'[zygote] start failed: {exc}; using direct encoder', file=sys.stderr, flush=True)


def wait_pidfd(fd):
    while True:
        try:
            if select.select([fd], [], [], None)[0]:
                return
        except InterruptedError:
            pass


def client_run(a, conn):
    """None means no worker was started: safe to use the direct encoder."""
    pidfd = None
    transferred = False
    old_handlers = {}

    def cancel(signum, frame):
        try:
            send_message(conn, dict(op='cancel'))
        except OSError:
            pass

    try:
        request(conn, 'run', a, [sys.stdin.fileno(), sys.stdout.fileno()])
        transferred = True
        # A queued request waits without a startup deadline. The zygote must
        # return either retry (no fork) or a pidfd BEFORE releasing the child.
        conn.settimeout(None)
        _, fds, flags, _ = socket.recv_fds(conn, 1, 1)
        if flags & socket.MSG_CTRUNC:
            for fd in fds:
                os.close(fd)
            raise RuntimeError('truncated worker pidfd')
        if fds:
            pidfd = fds[0]
            os.set_inheritable(pidfd, False)
        msg = recv_message(conn)
        if msg['status'] == 'retry':
            return None
        if msg['status'] != 'started' or pidfd is None:
            raise RuntimeError('worker pidfd missing')
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            old_handlers[sig] = signal.signal(sig, cancel)
        send_message(conn, dict(op='go'))
        print(f"[zygote] worker={msg['pid']} daemon={msg['daemon_pid']}", file=sys.stderr, flush=True)
        result = recv_message(conn)
        if result.get('status') != 'exited':
            raise RuntimeError('invalid worker completion')
        wait_pidfd(pidfd)
        return result['code']
    except Exception as exc:
        if pidfd is not None:
            # Even if the daemon dies, do not let Maya reclaim memory while
            # this worker is still alive. Closing control cancels the worker.
            conn.close()
            wait_pidfd(pidfd)
        if transferred:
            print('ERR zygote: ' + str(exc).replace('\n', ' ')[:300], flush=True)
            return 1
        return None
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        if pidfd is not None:
            os.close(pidfd)
        conn.close()


def model_signature(a):
    """Only stat files actually used by vision, not all text-model shards."""
    ckpt = pathlib.Path(a.mmproj)
    index = ckpt / 'model.safetensors.index.json'
    wmap = json.loads(index.read_text())['weight_map']
    files = {SCRIPT, ckpt / 'config.json', index, ckpt / 'preprocessor_config.json',
             pathlib.Path(HF) / 'modeling_mimo_v2.py', pathlib.Path(HF) / 'configuration_mimo_v2.py'}
    files.update(ckpt / shard for name, shard in wmap.items() if name.startswith('visual.'))
    return tuple((str(p), s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
                 for p in sorted(files) for s in [p.stat()])


def worker_main(a, prepared, input_fd, output_fd, gate_fd, parent_pid, prctl):
    # Install the Linux parent-death signal before waiting for client ACK.
    # The getppid check closes the fork/prctl race.
    if prctl(1, signal.SIGKILL, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        os._exit(1)
    if os.getppid() != parent_pid:
        os._exit(1)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, signal.SIG_DFL)
    os.dup2(input_fd, 0)
    os.dup2(output_fd, 1)
    # Do not retain listener/lock/control/other workers' pipe descriptors.
    for name in os.listdir('/proc/self/fd'):
        fd = int(name)
        if fd > 2 and fd != gate_fd:
            try:
                os.close(fd)
            except OSError:
                pass
    sys.stdin = os.fdopen(0, 'r', encoding='utf-8')
    sys.stdout = os.fdopen(1, 'w', encoding='utf-8', buffering=1)
    code = 1
    try:
        if os.read(gate_fd, 1) != b'G':
            os._exit(1)
        os.close(gate_fd)
        code = run_encoder(a, prepared)
    except BaseException as exc:
        try:
            print('ERR ' + str(exc).replace('\n', ' ')[:300], flush=True)
        except (BrokenPipeError, OSError):
            pass
    finally:
        try:
            sys.stdout.flush()
        except (BrokenPipeError, OSError):
            pass
    # No Python finalizers or inherited threading/atexit callbacks after fork.
    os._exit(code)


def zygote_main(a):
    path, lockpath = socket_paths(a)
    lockfd = os.open(lockpath, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(lockfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(lockfd)
        return 0
    listener = None
    children = {}
    sel = selectors.DefaultSelector()
    stopping = [False]
    stop_clients = []
    old_handlers = {}
    bound = False

    def stop(signum, frame):
        stopping[0] = True

    def retire():
        nonlocal bound
        stopping[0] = True
        if listener is not None and bound:
            sel.unregister(listener)
            listener.close()
            os.unlink(path)
            bound = False

    def close_child(state):
        for key in ('conn', 'pidfd'):
            try:
                sel.unregister(state[key])
            except (KeyError, ValueError):
                pass
        state['conn'].close()
        os.close(state['pidfd'])
        if state['gate'] is not None:
            os.close(state['gate'])

    try:
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            old_handlers[sig] = signal.signal(sig, stop)
        record = dict(pid=os.getpid(), identity=identity(a),
                      start_ticks=pathlib.Path('/proc/self/stat').read_text().split(') ')[1].split()[19])
        os.ftruncate(lockfd, 0)
        os.write(lockfd, json.dumps(record).encode())
        signature = model_signature(a)
        # Lazy torch/transformers imports and loading happen exactly once.
        # There is deliberately no torch.cuda.* call anywhere in this parent.
        # Limit OpenMP BEFORE the imports as well as before the fork.
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        load_runtime()
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        tower, cfg = load_tower(pathlib.Path(a.mmproj), 'cpu',
                               torch.bfloat16 if a.gpu else torch.float32)
        proc, merge = make_processor(pathlib.Path(a.mmproj), a.max_tokens)
        if torch.cuda.__dict__.get('_initialized', False):
            raise RuntimeError('CUDA was initialized in the zygote; refusing to fork')
        prepared = tower, cfg, proc, merge
        if model_signature(a) != signature or stopping[0]:
            return 0
        # Resolve libc before fork; no additional import machinery in children.
        prctl = ctypes.CDLL(None, use_errno=True).prctl
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        if os.path.lexists(path):
            os.unlink(path)  # only the flock owner can remove a stale endpoint
        listener.bind(path)
        bound = True
        os.chmod(path, 0o600)
        listener.listen(32)
        sel.register(listener, selectors.EVENT_READ, 'listener')
        last_activity = time.monotonic()
        next_check = last_activity
        print(f'[zygote] ready pid={os.getpid()} socket={path} dtype={tower.dtype}', flush=True)
        while True:
            now = time.monotonic()
            if not stopping[0] and now >= next_check:
                try:
                    stale = model_signature(a) != signature
                except (OSError, ValueError, KeyError):
                    stale = True
                next_check = now + 1
                if stale or (not children and now - last_activity >= a.zygote_idle_timeout):
                    stopping[0] = True
            if stopping[0]:
                retire()
                if not children:
                    break
            for key, _ in sel.select(timeout=0.2):
                if key.data == 'listener':
                    conn, _ = listener.accept()
                    conn.settimeout(2)
                    fds = []
                    try:
                        marker, fds, flags, _ = socket.recv_fds(conn, 1, 2)
                        if marker != b'V' or flags & socket.MSG_CTRUNC:
                            raise ValueError('bad fd transfer')
                        req = recv_message(conn)
                        if req.get('identity') != identity(a):
                            raise ValueError('argument set mismatch')
                        op = req.get('op')
                        if op == 'stop':
                            stopping[0] = True
                            stop_clients.append(conn)
                            conn = None
                        elif op == 'status':
                            send_message(conn, dict(status='ready', pid=os.getpid(), workers=list(children),
                                                    dtype=str(tower.dtype), threads=torch.get_num_threads()))
                        elif op == 'run' and len(fds) == 2:
                            # Recheck immediately, even between periodic checks.
                            try:
                                fresh = model_signature(a) == signature
                            except (OSError, ValueError, KeyError):
                                fresh = False
                            if not fresh or stopping[0]:
                                socket.send_fds(conn, [b'V'], [])
                                send_message(conn, dict(status='retry'))
                                stopping[0] = True
                            else:
                                read_gate, write_gate = os.pipe()
                                parent_pid = os.getpid()
                                try:
                                    pid = os.fork()
                                except OSError:
                                    os.close(read_gate)
                                    os.close(write_gate)
                                    raise
                                if pid == 0:
                                    worker_main(a, prepared, *fds, read_gate, parent_pid, prctl)
                                os.close(read_gate)
                                try:
                                    pidfd = os.pidfd_open(pid)
                                except OSError:
                                    os.kill(pid, signal.SIGKILL)
                                    os.waitpid(pid, 0)
                                    os.close(write_gate)
                                    raise
                                state = dict(pid=pid, conn=conn, pidfd=pidfd, gate=write_gate)
                                children[pid] = state
                                sel.register(pidfd, selectors.EVENT_READ, ('dead', pid))
                                sel.register(conn, selectors.EVENT_READ, ('control', pid))
                                # A worker cannot initialize CUDA until the client
                                # owns this pidfd and acknowledges it with 'go'.
                                socket.send_fds(conn, [b'V'], [pidfd])
                                send_message(conn, dict(status='started', pid=pid, daemon_pid=os.getpid()))
                                conn = None
                                last_activity = time.monotonic()
                        else:
                            raise ValueError('invalid zygote operation or fd count')
                    except Exception as exc:
                        print(f'[zygote] request error: {exc}', file=sys.stderr, flush=True)
                        for state in children.values():
                            if state['conn'] is conn:
                                os.kill(state['pid'], signal.SIGKILL)
                    finally:
                        for fd in fds:
                            os.close(fd)
                        if conn is not None:
                            conn.close()
                else:
                    kind, pid = key.data
                    state = children.get(pid)
                    if state is None:
                        continue
                    if kind == 'control':
                        try:
                            msg = recv_message(state['conn'])
                            if msg.get('op') != 'go' or state['gate'] is None:
                                raise ValueError('cancelled')
                            os.write(state['gate'], b'G')
                            os.close(state['gate'])
                            state['gate'] = None
                        except (OSError, EOFError, ValueError):
                            os.kill(pid, signal.SIGKILL)
                            sel.unregister(state['conn'])
                    elif kind == 'dead':
                        _, status = os.waitpid(pid, 0)
                        code = os.waitstatus_to_exitcode(status)
                        if code < 0:
                            code = 128 - code
                        try:
                            send_message(state['conn'], dict(status='exited', code=code))
                        except OSError:
                            pass
                        close_child(state)
                        del children[pid]
                        last_activity = time.monotonic()
        return 0
    finally:
        # Graceful stop drains existing workers. Unexpected daemon errors kill
        # and reap its own children; PDEATHSIG also covers SIGKILL/crashes.
        for pid, state in children.items():
            try:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
            except ProcessLookupError:
                pass
            close_child(state)
        if bound:
            listener.close()
            os.unlink(path)
        for conn in stop_clients:
            try:
                send_message(conn, dict(status='stopped', pid=os.getpid()))
            except OSError:
                pass
            conn.close()
        sel.close()
        os.close(lockfd)
        # Keep the lock inode. Unlinking it races flock waiters.
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        print(f'[zygote] exit pid={os.getpid()}', flush=True)


def starting_pidfd(a):
    """Find only this script/argument set's flock owner during heavy imports."""
    try:
        lockfd = os.open(socket_paths(a)[1], os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return None
    try:
        try:
            fcntl.flock(lockfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return None  # old metadata without a live owner is never trusted
        except BlockingIOError:
            pass
        record = None
        for _ in range(10):
            try:
                record = json.loads(os.pread(lockfd, 65536, 0))
                break
            except ValueError:
                time.sleep(0.02)
        if record is None or record.get('identity') != identity(a):
            return None
        pid = record['pid']
        fd = os.pidfd_open(pid)
        try:
            proc = pathlib.Path(f'/proc/{pid}')
            ticks = (proc / 'stat').read_text().split(') ')[1].split()[19]
            cmd = (proc / 'cmdline').read_bytes().split(b'\0')
            if ticks != record['start_ticks'] or os.fsencode(SCRIPT) not in cmd or b'--zygote' not in cmd:
                os.close(fd)
                return None
            return pid, fd
        except BaseException:
            os.close(fd)
            raise
    except (ProcessLookupError, FileNotFoundError):
        return None
    finally:
        os.close(lockfd)


def control(a, op):
    conn = connect(a)
    if conn is None:
        owner = starting_pidfd(a)
        if owner is None:
            print(json.dumps(dict(status='absent')))
            return 0 if op == 'stop' else 1
        pid, fd = owner
        try:
            if op == 'stop':
                signal.pidfd_send_signal(fd, signal.SIGTERM)
                wait_pidfd(fd)
                status = 'stopped'
            else:
                status = 'starting'
            print(json.dumps(dict(status=status, pid=pid)))
        finally:
            os.close(fd)
        return 0 if op == 'stop' else 1
    with conn:
        request(conn, op, a)
        conn.settimeout(None if op == 'stop' else 3)
        result = recv_message(conn)
        print(json.dumps(result, sort_keys=True), flush=True)
    if op == 'stop':
        try:
            fd = os.pidfd_open(result['pid'])
        except ProcessLookupError:
            pass
        else:
            try:
                wait_pidfd(fd)
            finally:
                os.close(fd)
    return 0


def main():
    a = parse_args()
    if a.zygote:
        return zygote_main(a)
    if a.zygote_stop:
        return control(a, 'stop')
    if a.zygote_status:
        return control(a, 'status')
    if not (a.measure or a.direct):
        try:
            conn = connect(a)
            if conn is not None:
                code = client_run(a, conn)
                if code is not None:
                    return code
            spawn_zygote(a)
        except (OSError, ValueError) as exc:
            print(f'[zygote] unavailable: {exc}; using direct encoder', file=sys.stderr, flush=True)
    try:
        return run_encoder(a)
    except Exception as exc:
        print('ERR ' + str(exc).replace('\n', ' ')[:300], flush=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
