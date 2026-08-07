import json
import os
from filelock import FileLock

REPO_FILE = "data/repositories.json"

LOCK_FILE = "data/repositories.lock"


def load_repositories():
    if not os.path.exists(REPO_FILE):
        return []

    with open(REPO_FILE, "r") as f:
        return json.load(f)


def save_repository(repo_name):
    with FileLock(LOCK_FILE):
      repos = load_repositories()

      if repo_name not in repos:
        repos.append(repo_name)

        with open(REPO_FILE, "w") as f:
           json.dump(repos, f, indent=4)

def delete_repository(repo_name):
    with FileLock(LOCK_FILE):
        repos = load_repositories()

        if repo_name not in repos:
            return False

        repos.remove(repo_name)

        with open(REPO_FILE, "w") as f:
            json.dump(repos, f, indent=4)

        return True
