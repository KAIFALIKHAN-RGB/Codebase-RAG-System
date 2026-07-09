from src.embeddings.embedder import get_embedding

vector = get_embedding("def login(username, password):")

print("Vector Dimension : ", len(vector))
print("First 10 elements of the vector : ")
print(vector[:10])