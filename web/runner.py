"""统一通过 subprocess 跑后端脚本"""

import subprocess
import sys
import os
import signal
from pathlib import Path
from typing import Generator, Optional


# 项目根目录（web/ 的上一级）
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
PYTHON = sys.executable


class ProcessRunner:
    """把 main.py / train_contrastive.py / llm_data_generator 起在子进程里"""

    def __init__(self):
        self.process: Optional[subprocess.Popen] = None

    @property
    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def run_main_flow(self, **kwargs) -> Generator[str, None, None]:
        cmd = self._build_cmd([PYTHON, "main.py"], kwargs)
        yield from self._execute(cmd)

    def run_contrastive(self, **kwargs) -> Generator[str, None, None]:
        cmd = self._build_cmd([PYTHON, "train_contrastive.py"], kwargs)
        yield from self._execute(cmd)

    def run_llm_gen(self, **kwargs) -> Generator[str, None, None]:
        cmd = self._build_cmd([PYTHON, "-m", "llm_data_generator.main"], kwargs)
        yield from self._execute(cmd)

    def stop(self):
        if self.process and self.process.poll() is None:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            except (ProcessLookupError, OSError):
                self.process.terminate()
            self.process = None

    def _build_cmd(self, base: list, params: dict) -> list:
        cmd = list(base)
        for key, value in params.items():
            flag = "--" + key.replace("_", "-")
            if isinstance(value, bool):
                cmd.append(flag if value else f"--no-{key.replace('_', '-')}")
            elif value is not None and value != "":
                cmd.extend([flag, str(value)])
        return cmd

    def _execute(self, cmd: list) -> Generator[str, None, None]:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=PROJECT_ROOT,
            text=True,
            bufsize=1,
            env=env,
            preexec_fn=os.setsid,
        )

        for line in self.process.stdout:
            yield line

        self.process.wait()
        rc = self.process.returncode
        self.process = None
        if rc != 0:
            yield f"\n[Process exited with code {rc}]\n"
