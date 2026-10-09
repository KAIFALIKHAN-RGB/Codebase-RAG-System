import json
import os

from filelock import FileLock


REPO_PATH_FILE = "data/repository_paths.json"
LOCK_FILE = "data/repository_paths.lock"


def load_repository_paths():
    if not os.path.exists(REPO_PATH_FILE):
        return {}

    with open(REPO_PATH_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_repository_path(repo_name, repo_path):
    canonical_path = os.path.realpath(repo_path)

    if not os.path.isdir(canonical_path):
        raise ValueError("Repository path must be an existing directory.")

    os.makedirs(os.path.dirname(LOCK_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(REPO_PATH_FILE), exist_ok=True)

    with FileLock(LOCK_FILE):
        paths = load_repository_paths()
        paths[repo_name] = canonical_path

        with open(REPO_PATH_FILE, "w", encoding="utf-8") as f:
            json.dump(paths, f, indent=4)


def get_repository_path(repo_name):
    paths = load_repository_paths()
    return paths.get(repo_name)


def delete_repository_path(repo_name):
    with FileLock(LOCK_FILE):
        paths = load_repository_paths()

        if repo_name not in paths:
            return False

        del paths[repo_name]

        with open(
            REPO_PATH_FILE,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(paths, f, indent=4)

    return True