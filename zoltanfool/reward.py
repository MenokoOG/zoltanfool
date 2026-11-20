"""Stage 6. Reward is 1 if every test passes, else 0. Model output is untrusted code."""
import math
import os
import subprocess
import sys
import tempfile
from types import ModuleType

resource: ModuleType | None
try:
    import resource
except ImportError:                                    # not available on Windows
    resource = None


def _limits() -> None:
    """Runs in the child before the program starts. Caps memory and CPU seconds."""
    assert resource is not None
    resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))     # 1 GiB address space
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))  # files up to 1 MiB


def reward(prompt: str, completion: str, tests: str, timeout: float = 5.0) -> int:
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        raise RuntimeError("refusing to run generated code as root")
    program = prompt + completion + "\n\n" + tests + "\n"
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "attempt.py")
        with open(path, "w") as f:
            f.write(program)
        try:
            r = subprocess.run([sys.executable, "-I", path], cwd=tmp, timeout=timeout,
                               capture_output=True, text=True, env={},
                               preexec_fn=_limits if resource else None)
        except subprocess.TimeoutExpired:
            return 0
        return int(r.returncode == 0)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimate: chance at least one of k samples passes, given c of n passed."""
    if k > n:
        raise ValueError(f"pass@{k} needs at least {k} samples, got n={n}")
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)
