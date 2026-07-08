import os
import sys
from pathlib import Path


def repo_root():
    return Path(__file__).resolve().parents[1]


def add_pidsmaker_src():
    candidates = []
    src_env = os.environ.get("PIDSMAKER_SRC")
    root_env = os.environ.get("PIDSMAKER_ROOT")
    if src_env:
        candidates.append(Path(src_env).expanduser())
    if root_env:
        candidates.append(Path(root_env).expanduser() / "src")
    candidates.extend([
        Path("/home/src"),
        repo_root() / "external" / "PIDSMaker" / "src",
    ])

    for path in candidates:
        if path.exists():
            resolved = str(path.resolve())
            if resolved not in sys.path:
                sys.path.insert(0, resolved)
            return path

    raise FileNotFoundError(
        "PIDSMaker source not found. Run `git submodule update --init --recursive`, "
        "or set PIDSMAKER_SRC/PIDSMAKER_ROOT."
    )


def artifacts_root():
    env_root = os.environ.get("ARTIFACTS_DIR")
    if env_root:
        return Path(env_root).expanduser()

    candidates = [
        Path("/home/artifacts"),
        repo_root() / "artifacts",
        repo_root().parent / "PIDSMaker-velox-paper" / "artifacts",
    ]
    for path in candidates:
        if path.exists():
            return path
    return repo_root() / "artifacts"
