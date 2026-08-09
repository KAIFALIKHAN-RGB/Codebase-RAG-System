import json
from pathlib import Path
from filelock import FileLock

STATE_FILE = Path("data/file_hashes.json")
LOCK_FILE = Path("data/file_hashes.lock")

def load_index_state():
    if not STATE_FILE.exists():
        return {}
    
    with open(STATE_FILE, "r", encoding="utf-8") as file:
        return json.load(file)

def save_index_state(state):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

    with FileLock(LOCK_FILE):
      with open(STATE_FILE, "w", encoding="utf-8") as file:
          json.dump(state, file, indent=4)

def delete_repository_state(repo_name):
    with FileLock(LOCK_FILE):  
      state = load_index_state()

      prefix = repo_name + "\\"

      new_state = {
        file_path: file_hash
        for file_path, file_hash in state.items()
        if not file_path.startswith(prefix)
      }

    with open(STATE_FILE, "w", encoding="utf-8") as file:
        json.dump(new_state, file, indent=4)