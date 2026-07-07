import json

def save_chunks(chunks, output_path):
    with open(output_path, "w" , encoding="utf-8") as file:
        json.dump(chunks, file, indent=4, ensure_ascii=False)