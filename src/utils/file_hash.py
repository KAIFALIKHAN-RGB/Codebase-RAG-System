import hashlib

def get_hash_file(file_path):
    with open(file_path, "rb") as file:
        content = file.read()

    return hashlib.sha256(content).hexdigest()