import json
from pathlib import Path

STATE_FILE = Path("data/file_hashes.json")

def load_index_state():
    if not STATE_FILE.exists():
        return {}
    
    with open(STATE_FILE, "r", encoding="utf-8") as file:
        return json.load(file)

def save_index_state(state):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    with open(STATE_FILE, "w", encoding="utf-8") as file:
        json.dump(state, file, indent=4)