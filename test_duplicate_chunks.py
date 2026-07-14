from src.storage.chroma_store import collection

data = collection.get()

seen = {}
duplicates = []

for i, metadata in enumerate(data["metadatas"]):
    key = (
        metadata.get("file_path"),
        metadata.get("name"),
        metadata.get("start_line"),
        metadata.get("end_line")
    )

    if key in seen:
        duplicates.append(key)
    else:
        seen[key] = i

print("Total records:", len(data["ids"]))
print("Unique chunks:", len(seen))
print("Duplicate chunks:", len(duplicates))

if duplicates:
    print("\nDuplicates found:")
    for duplicate in duplicates:
        print(duplicate)
else:
    print("\nNo duplicate chunks found.")