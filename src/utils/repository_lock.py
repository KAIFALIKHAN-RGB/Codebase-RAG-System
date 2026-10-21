import os

from filelock import FileLock


LOCK_DIR = "data/repo_locks"


def get_repository_lock(repo_name: str) -> FileLock:
    if (
        not repo_name
        or repo_name in {".", ".."}
        or os.path.basename(repo_name) != repo_name
        or "/" in repo_name
        or "\\" in repo_name
    ):
        raise ValueError("Invalid repository name.")

    os.makedirs(LOCK_DIR, exist_ok=True)
    return FileLock(os.path.join(LOCK_DIR, f"{repo_name}.lock"))
