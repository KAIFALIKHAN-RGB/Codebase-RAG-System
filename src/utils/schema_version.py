import json 
import os

SCHEMA_VERSION = 1
SCHEMA_VERSION_FILE = "data/schema_version.json"

def load_schema_version():
  if not os.path.exists(SCHEMA_VERSION_FILE):
    return None
  with open(SCHEMA_VERSION_FILE, "r", encoding="utf-8") as file:
    data = json.load(file)
  return data["schema_version"]


def save_schema_version():
  with open(SCHEMA_VERSION_FILE, "w", encoding="utf-8") as file:
    json.dump({"schema_version": SCHEMA_VERSION}, file, indent=4)
 