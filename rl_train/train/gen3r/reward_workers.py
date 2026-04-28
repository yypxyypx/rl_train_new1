"""reward_workers.py — 跨 conda env 的常驻 reward worker 进程池。

设计目标：
  - 每个 reward 模型（DA3 / DINOv2 / VideoAlign）只加载一次，整个训练过程复用
  - 跨 conda env 隔离（worker 在自己的 env 子进程里运行）
  - IPC 走 stdin/stdout JSON line 协议，不传 Python 对象、不传 GPU tensor
  - 中间结果（npz/json）落 /dev/shm（tmpfs），主进程 np.load 读回，~200ms 来回

通信协议（每行一个 JSON）：
  父 → 子:
    {"cmd": "ping"}                             启动握手，等待 "ready"
    {"cmd": "run", "job_id": k, "args": {...}}  执行一次推理
    {"cmd": "exit"}                             关闭

  子 → 父:
    {"status": "ready"}                                                   模型加载完成
    {"job_id": k, "status": "ok",     "output_path": "...", "elapsed": 25.3}
    {"job_id": k, "status": "FAILED", "error": "..."}

主类：
  - WorkerProcess: 一个常驻 Popen 子进程的封装
  - RewardWorkerPool: 三个 worker 的集合，提供 submit / wait / shutdown
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from queue import Queue, Empty
from subprocess import Popen, PIPE
from typing import Optional

_HERE = Path(__file__).resolve().parent

# 每个 worker 类型对应的 conda env
WORKER_CONDA_ENV = {
    "da3":        "rl_da3",
    "dinov2":     "rl_da3",
    "videoalign": "Videoalign",
}

# /dev/shm 默认 IPC 目录（tmpfs，速度近内存）
DEFAULT_SHM_DIR = "/dev/shm/gen3r_reward"


def _env_python(env_name: str) -> str:
    candidates = [
        f"/opt/conda/envs/{env_name}/bin/python",
        os.path.expanduser(f"~/miniconda3/envs/{env_name}/bin/python"),
        os.path.expanduser(f"~/anaconda3/envs/{env_name}/bin/python"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(
        f"Cannot find python for conda env '{env_name}'. Tried: {candidates}"
    )


class WorkerError(RuntimeError):
    """worker 子进程报告的错误（区别于本地代码异常）。"""
    pass


class WorkerProcess:
    """常驻子进程封装：在指定 conda env 中跑 reward_workers_main.py 的某个 worker。

    用法：
        w = WorkerProcess("da3", gpu_id=0)
        w.start()                                  # 阻塞等待模型加载完成
        out = w.run({"frames_dir": "...", "output_path": "..."}, timeout=300)
        w.shutdown()
    """

    def __init__(
        self,
        worker_type: str,
        gpu_id: int = 0,
        log_file: Optional[str] = None,
        extra_env: Optional[dict] = None,
        ready_timeout: float = 600.0,
    ):
        if worker_type not in WORKER_CONDA_ENV:
            raise ValueError(
                f"Unknown worker_type {worker_type!r}. Valid: {list(WORKER_CONDA_ENV)}"
            )
        self.worker_type = worker_type
        self.env_name = WORKER_CONDA_ENV[worker_type]
        self.gpu_id = gpu_id
        self.log_file = log_file
        self.extra_env = extra_env or {}
        self.ready_timeout = ready_timeout

        self.proc: Optional[Popen] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._stdout_thread: Optional[threading.Thread] = None
        self._stdout_q: Queue = Queue()
        self._stderr_log = None
        self._lock = threading.Lock()  # 保护 stdin

    # ────────────────────────────────────────────────────────────────────
    # 启动 / 关闭
    # ────────────────────────────────────────────────────────────────────

    def start(self) -> None:
        """spawn 子进程并阻塞等待 ready。失败抛 RuntimeError。

        若想并行启动多个 worker，调用方应先对每个 WorkerProcess 调
        ``spawn()``（只 Popen + drain，不等 ready），再统一调 ``wait_ready()``。
        """
        self.spawn()
        self.wait_ready()

    def spawn(self) -> None:
        """只启动子进程并 send ping，不等 ready。"""
        if self.proc is not None:
            raise RuntimeError(f"WorkerProcess[{self.worker_type}] already started")

        py = _env_python(self.env_name)
        script = str(_HERE / "reward_workers_main.py")
        cmd = [py, "-u", script, "--worker", self.worker_type]

        # 子进程环境变量：
        #   - 继承父进程
        #   - CUDA_VISIBLE_DEVICES 按 gpu_id 切片
        #   - 子进程 import 可能要用到的额外路径
        env = dict(os.environ)
        parent_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if parent_cvd:
            visible = [d.strip() for d in parent_cvd.split(",") if d.strip()]
            if 0 <= self.gpu_id < len(visible):
                env["CUDA_VISIBLE_DEVICES"] = visible[self.gpu_id]
            else:
                env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        else:
            env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        # PyTorch alloc 配置：worker 进程独立显存池，让它复用父侧设置
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        env.update({k: str(v) for k, v in self.extra_env.items()})

        if self.log_file:
            os.makedirs(os.path.dirname(self.log_file) or ".", exist_ok=True)
            self._stderr_log = open(self.log_file, "w", buffering=1)
        else:
            self._stderr_log = None

        self.proc = Popen(
            cmd,
            stdin=PIPE,
            stdout=PIPE,
            stderr=PIPE,
            env=env,
            cwd=str(_HERE),
            text=True,
            bufsize=1,  # 行缓冲
        )

        # 后台线程：drain stderr → 日志文件（避免 PIPE 满死锁）
        def _drain_stderr():
            assert self.proc is not None
            for line in self.proc.stderr:
                if self._stderr_log:
                    self._stderr_log.write(line)
                else:
                    sys.stderr.write(f"[worker-{self.worker_type}] {line}")

        self._stderr_thread = threading.Thread(
            target=_drain_stderr,
            name=f"stderr-{self.worker_type}",
            daemon=True,
        )
        self._stderr_thread.start()

        # 后台线程：drain stdout → Queue，主线程靠 Queue.get(timeout=…) 读
        def _drain_stdout():
            assert self.proc is not None
            for line in self.proc.stdout:
                self._stdout_q.put(line.rstrip("\n"))
            self._stdout_q.put(None)  # EOF 信号

        self._stdout_thread = threading.Thread(
            target=_drain_stdout,
            name=f"stdout-{self.worker_type}",
            daemon=True,
        )
        self._stdout_thread.start()

        # 发 ping（不等 ready，由 wait_ready 处理）
        self._send({"cmd": "ping"})

    def wait_ready(self) -> None:
        """阻塞等子进程报告 ready。spawn() 之后调用。"""
        assert self.proc is not None, "must call spawn() first"
        deadline = time.time() + self.ready_timeout
        while time.time() < deadline:
            line = self._read_line(timeout=5.0)
            if line is None:
                if self.proc.poll() is not None:
                    raise RuntimeError(
                        f"WorkerProcess[{self.worker_type}] died during startup "
                        f"(exit_code={self.proc.returncode}). Check log: {self.log_file}"
                    )
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                # worker 启动期间可能有一些非 JSON 行（库的警告等），跳过
                if self._stderr_log:
                    self._stderr_log.write(f"[stdout-noise] {line}\n")
                continue
            if msg.get("status") == "ready":
                print(f"[WorkerPool] {self.worker_type} ready "
                      f"(loaded in {msg.get('elapsed', 0):.1f}s)")
                return
            if msg.get("status") == "FAILED":
                raise RuntimeError(
                    f"WorkerProcess[{self.worker_type}] startup failed: "
                    f"{msg.get('error', 'unknown')}"
                )
        raise TimeoutError(
            f"WorkerProcess[{self.worker_type}] did not become ready in "
            f"{self.ready_timeout}s. Check log: {self.log_file}"
        )

    def shutdown(self, timeout: float = 30.0) -> None:
        """优雅关闭子进程；超时则 kill。"""
        if self.proc is None or self.proc.poll() is not None:
            return
        try:
            self._send({"cmd": "exit"})
        except (BrokenPipeError, OSError):
            pass
        try:
            self.proc.wait(timeout=timeout)
        except Exception:
            print(f"[WorkerPool] {self.worker_type} did not exit in {timeout}s, killing")
            self.proc.kill()
            self.proc.wait(timeout=5.0)
        finally:
            if self._stderr_log:
                try:
                    self._stderr_log.close()
                except Exception:
                    pass

    # ────────────────────────────────────────────────────────────────────
    # 同步 RPC：run 一次，等结果
    # ────────────────────────────────────────────────────────────────────

    def run(self, args: dict, timeout: float = 600.0) -> dict:
        """同步执行一个 job，返回 worker 响应 dict（含 output_path / elapsed）。

        失败抛 WorkerError。子进程崩溃抛 RuntimeError。
        """
        if self.proc is None or self.proc.poll() is not None:
            raise RuntimeError(
                f"WorkerProcess[{self.worker_type}] is not alive "
                f"(exit_code={self.proc.returncode if self.proc else None})"
            )
        job_id = uuid.uuid4().hex[:12]
        self._send({"cmd": "run", "job_id": job_id, "args": args})

        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self._read_line(timeout=5.0)
            if line is None:
                if self.proc.poll() is not None:
                    raise RuntimeError(
                        f"WorkerProcess[{self.worker_type}] died during job {job_id}"
                    )
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                if self._stderr_log:
                    self._stderr_log.write(f"[stdout-noise] {line}\n")
                continue
            if msg.get("job_id") != job_id:
                # 不应该出现（worker 串行处理），但防御性跳过
                if self._stderr_log:
                    self._stderr_log.write(f"[unexpected job_id] {line}\n")
                continue
            if msg.get("status") == "ok":
                return msg
            else:
                raise WorkerError(
                    f"[{self.worker_type}] job {job_id} FAILED: "
                    f"{msg.get('error', 'unknown')}"
                )
        raise TimeoutError(
            f"WorkerProcess[{self.worker_type}] job {job_id} timeout after {timeout}s"
        )

    def run_batch(self, sub_jobs: list, *, extra_args: Optional[dict] = None,
                  timeout: float = 1800.0) -> dict:
        """同步在 worker 上批量执行 N 个 sub-jobs（worker 内部串行或真 batch）。

        参数:
            sub_jobs   : list[dict]，每个 dict 含 worker 需要的 args
                          (如 DA3/DINO 用 {frames_dir, output_path}；
                           VideoAlign 用 {video_path, prompt, output_path})
            extra_args : 顶层附加参数（如 DINO 的 forward_batch=8）
            timeout    : 整批的超时时长（秒）

        返回 worker 响应 dict，含 'results': list[dict]，每个 dict 至少含
        output_path/status/elapsed。
        """
        if self.proc is None or self.proc.poll() is not None:
            raise RuntimeError(
                f"WorkerProcess[{self.worker_type}] is not alive "
                f"(exit_code={self.proc.returncode if self.proc else None})"
            )
        job_id = uuid.uuid4().hex[:12]
        args_top: dict = {"jobs": sub_jobs}
        if extra_args:
            args_top.update(extra_args)
        self._send({"cmd": "run_batch", "job_id": job_id, "args": args_top})

        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self._read_line(timeout=5.0)
            if line is None:
                if self.proc.poll() is not None:
                    raise RuntimeError(
                        f"WorkerProcess[{self.worker_type}] died during batch {job_id}"
                    )
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                if self._stderr_log:
                    self._stderr_log.write(f"[stdout-noise] {line}\n")
                continue
            if msg.get("job_id") != job_id:
                if self._stderr_log:
                    self._stderr_log.write(f"[unexpected job_id] {line}\n")
                continue
            if msg.get("status") == "ok":
                return msg
            else:
                raise WorkerError(
                    f"[{self.worker_type}] batch {job_id} FAILED "
                    f"(completed {len(msg.get('results', []))}/{len(sub_jobs)} "
                    f"sub-jobs): {msg.get('error', 'unknown')}"
                )
        raise TimeoutError(
            f"WorkerProcess[{self.worker_type}] batch {job_id} timeout after {timeout}s"
        )

    # ────────────────────────────────────────────────────────────────────
    # 同步 RPC：unload / reload（GPU↔CPU 模型迁移）
    # ────────────────────────────────────────────────────────────────────

    def _wait_status(self, expected: str, timeout: float) -> dict:
        """阻塞等下一条 status == expected 的响应（忽略带 job_id 的 run 响应）。"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self._read_line(timeout=2.0)
            if line is None:
                if self.proc is not None and self.proc.poll() is not None:
                    raise RuntimeError(
                        f"WorkerProcess[{self.worker_type}] died during {expected}")
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                if self._stderr_log:
                    self._stderr_log.write(f"[stdout-noise] {line}\n")
                continue
            # unload/reload 响应不带 job_id；带 job_id 的是积压的 run 响应（理论上不应该出现）
            if "job_id" in msg:
                if self._stderr_log:
                    self._stderr_log.write(f"[stale job response] {line}\n")
                continue
            if msg.get("status") == expected:
                return msg
            if msg.get("status") == "FAILED":
                raise WorkerError(
                    f"[{self.worker_type}] {expected} FAILED: {msg.get('error','unknown')}")
        raise TimeoutError(
            f"WorkerProcess[{self.worker_type}] {expected} timeout after {timeout}s")

    def unload(self, timeout: float = 30.0) -> None:
        """让 worker 把模型从 GPU 移到 CPU，释放显存。"""
        if self.proc is None or self.proc.poll() is not None:
            return
        self._send({"cmd": "unload"})
        self._wait_status("unloaded", timeout)

    def reload(self, timeout: float = 60.0) -> None:
        """让 worker 把模型搬回 GPU，下次 run 之前必须调。"""
        if self.proc is None or self.proc.poll() is not None:
            return
        self._send({"cmd": "reload"})
        self._wait_status("ready", timeout)

    # ────────────────────────────────────────────────────────────────────
    # 内部：收发
    # ────────────────────────────────────────────────────────────────────

    def _send(self, msg: dict) -> None:
        assert self.proc is not None and self.proc.stdin is not None
        line = json.dumps(msg, ensure_ascii=False) + "\n"
        with self._lock:
            self.proc.stdin.write(line)
            self.proc.stdin.flush()

    def _read_line(self, timeout: float = 1.0) -> Optional[str]:
        """从 stdout queue 读一行，超时返回 None。worker 退出（EOF）也返回 None。"""
        try:
            item = self._stdout_q.get(timeout=timeout)
        except Empty:
            return None
        return item  # None 表示 EOF


class RewardWorkerPool:
    """三个常驻 reward worker 的池子，提供 run(group, args) 同步 API。

    生命周期：
        pool = RewardWorkerPool(active_groups={"da3", "dinov2", "videoalign"}, gpu_id=0)
        pool.start()                    # 阻塞，等所有 worker ready
        out = pool.run("da3", {...})    # 同步 RPC
        pool.shutdown()
    """

    def __init__(
        self,
        active_groups: set[str],
        gpu_id: int = 0,
        log_dir: Optional[str] = None,
        ready_timeout: float = 600.0,
    ):
        unknown = active_groups - set(WORKER_CONDA_ENV.keys())
        if unknown:
            raise ValueError(
                f"Unknown worker types: {unknown}. Valid: {list(WORKER_CONDA_ENV)}"
            )
        self.active_groups = set(active_groups)
        self.gpu_id = gpu_id
        self.log_dir = log_dir
        self.ready_timeout = ready_timeout

        self.workers: dict[str, WorkerProcess] = {}
        self._started = False
        self._async_thread: Optional[threading.Thread] = None
        self._async_error: Optional[BaseException] = None
        self._async_done_evt = threading.Event()

        # 确保 SHM 工作目录存在
        os.makedirs(DEFAULT_SHM_DIR, exist_ok=True)

    def start(self) -> None:
        """spawn 所有 active worker（并行），阻塞等所有 ready。失败抛异常 + cleanup。

        三个 worker 并行加载（DA3 ~3min, DINOv2 ~30s, VideoAlign ~2min），
        总耗时 ~ max(...) ≈ 3min 而不是 sum(...) ≈ 5.5min。
        """
        if self._started:
            return
        try:
            # 1) 并行 spawn 所有 worker（只起进程，不等 ready）
            for group in sorted(self.active_groups):
                log_file = (
                    str(Path(self.log_dir) / f"worker_{group}.log")
                    if self.log_dir else None
                )
                w = WorkerProcess(
                    group, gpu_id=self.gpu_id,
                    log_file=log_file,
                    ready_timeout=self.ready_timeout,
                )
                w.spawn()
                self.workers[group] = w
                print(f"[WorkerPool] spawned {group} (env={w.env_name})")

            # 2) 顺序 wait_ready，但因为 spawn 是并行的，等的是 max() 而非 sum()
            for group, w in self.workers.items():
                w.wait_ready()
            self._started = True
            print(f"[WorkerPool] All workers ready: {list(self.workers.keys())}")
        except Exception:
            self.shutdown()
            raise

    def start_async(self) -> None:
        """异步在后台线程 spawn + wait_ready，立即返回。

        典型用法：
            pool.start_async()         # 立刻返回，主进程并行加载 Gen3R
            ...                        # 主进程做其他事情（加载模型、第一次 rollout）
            pool.wait_ready_join()     # 第一次需要 reward 之前阻塞确保 worker ready
            out = pool.run("da3", ...) # 也会自动 lazy join
        """
        if self._started or self._async_thread is not None:
            return

        def _bg():
            try:
                self.start()
            except BaseException as e:  # noqa: BLE001
                self._async_error = e
                print(f"[WorkerPool] async start failed: {e}")
            finally:
                self._async_done_evt.set()

        self._async_thread = threading.Thread(
            target=_bg, name="WorkerPoolStartAsync", daemon=True
        )
        self._async_thread.start()
        print(f"[WorkerPool] start_async() launched bg thread for "
              f"{sorted(self.active_groups)}")

    def wait_ready_join(self, timeout: Optional[float] = None) -> None:
        """阻塞直到 start_async() 完成。如果异步加载抛过异常会重新抛出。"""
        if self._started:
            return
        if self._async_thread is None:
            # 没用 start_async，直接同步 start
            self.start()
            return
        t0 = time.time()
        ok = self._async_done_evt.wait(timeout=timeout)
        if not ok:
            raise TimeoutError(
                f"WorkerPool async start did not finish within {timeout}s"
            )
        if self._async_error is not None:
            raise self._async_error
        dt = time.time() - t0
        if dt > 0.05:
            print(f"[WorkerPool] wait_ready_join() blocked {dt:.1f}s "
                  "for async worker startup")

    def is_ready(self) -> bool:
        return self._started

    def run(self, group: str, args: dict, timeout: float = 600.0) -> dict:
        """同步在指定 group 的 worker 上跑一个 job。如果使用了 start_async
        且后台还没 ready，这里会自动阻塞等待。"""
        if not self._started:
            if self._async_thread is not None:
                self.wait_ready_join()
            else:
                raise RuntimeError("RewardWorkerPool not started")
        if group not in self.workers:
            raise ValueError(f"Group {group!r} not active. Active: {list(self.workers)}")
        return self.workers[group].run(args, timeout=timeout)

    def run_batch(self, group: str, sub_jobs: list, *,
                  extra_args: Optional[dict] = None,
                  timeout: float = 1800.0) -> dict:
        """批量在指定 group 的 worker 上跑 N 个 sub-jobs。

        DA3       : 内部串行（不能跨 scene 真 batch）
        DINO      : 内部 forward 真 batch（默认 forward_batch=8，可在 extra_args 覆盖）
        VideoAlign: 一次 inferencer.reward(list, list) 真 batch
        """
        if not self._started:
            if self._async_thread is not None:
                self.wait_ready_join()
            else:
                raise RuntimeError("RewardWorkerPool not started")
        if group not in self.workers:
            raise ValueError(f"Group {group!r} not active. Active: {list(self.workers)}")
        return self.workers[group].run_batch(
            sub_jobs, extra_args=extra_args, timeout=timeout,
        )

    # ────────────────────────────────────────────────────────────────────
    # 显存管理：把所有 worker 模型暂存到 CPU / 搬回 GPU
    # ────────────────────────────────────────────────────────────────────

    def unload_all(self, timeout: float = 30.0) -> float:
        """把所有 worker 上的模型从 GPU 移到 CPU，返回耗时（秒）。

        典型节省：DA3 ~7GiB + DINOv2 ~5GiB ≈ 12GiB（VideoAlign 0.8GiB no-op）。
        """
        if not self._started:
            return 0.0
        t0 = time.time()
        # 串行下发即可（每条 unload <1s）；并行的话需要线程，收益不大
        for group, w in self.workers.items():
            try:
                w.unload(timeout=timeout)
            except Exception as e:
                print(f"[WorkerPool] unload {group} failed: {e}")
        dt = time.time() - t0
        print(f"[WorkerPool] unload_all done in {dt:.2f}s "
              f"({list(self.workers.keys())})")
        return dt

    def reload_all(self, timeout: float = 60.0) -> float:
        """把所有 worker 上的模型搬回 GPU，返回耗时（秒）。"""
        if not self._started:
            return 0.0
        t0 = time.time()
        for group, w in self.workers.items():
            try:
                w.reload(timeout=timeout)
            except Exception as e:
                print(f"[WorkerPool] reload {group} failed: {e}")
        dt = time.time() - t0
        print(f"[WorkerPool] reload_all done in {dt:.2f}s "
              f"({list(self.workers.keys())})")
        return dt

    def shutdown(self) -> None:
        """关闭所有 worker，清理 SHM 残留文件。"""
        for group, w in list(self.workers.items()):
            try:
                w.shutdown()
            except Exception as e:
                print(f"[WorkerPool] error shutting down {group}: {e}")
        self.workers.clear()
        self._started = False

        # 清理 SHM 残留（只清 gen3r_reward 子目录下的文件）
        try:
            for f in Path(DEFAULT_SHM_DIR).glob("*"):
                try:
                    f.unlink()
                except Exception:
                    pass
        except Exception:
            pass
        print("[WorkerPool] Shutdown complete")

    # 上下文管理器，方便 with 包裹
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.shutdown()
        return False


def make_shm_path(step: int, rollout_idx: int, group: str, ext: str = "npz") -> str:
    """生成 /dev/shm 下的中间产物路径。"""
    os.makedirs(DEFAULT_SHM_DIR, exist_ok=True)
    return os.path.join(
        DEFAULT_SHM_DIR,
        f"step{step}_r{rollout_idx}_{group}.{ext}",
    )
