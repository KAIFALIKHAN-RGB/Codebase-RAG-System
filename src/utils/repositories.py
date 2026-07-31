import json
import os

REPO_FILE = "data/repositories.json"


def load_repositories():
    if not os.path.exists(REPO_FILE):
        return []

    with open(REPO_FILE, "r") as f:
        return json.load(f)


def save_repository(repo_name):
    repos = load_repositories()

    if repo_name not in repos:
        repos.append(repo_name)

    with open(REPO_FILE, "w") as f:
        json.dump(repos, f, indent=4)