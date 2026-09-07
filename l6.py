import faiss
import numpy as np

# Our stored vectors
vectors = np.array([
    [1.0, 1.0],  # Apple
    [1.2, 1.1],  # Mango
    [8.0, 9.0],  # Car
], dtype="float32")

# Create FAISS Index - 2 dimensions
index = faiss.IndexFlatL2(2)

# Add vectors
index.add(vectors)
print(f"Total vectors in index: {index.ntotal}")

# Search vectors
query = np.array([[1.1, 1.0]], dtype="float32")

# Find 2 nearest vectors
distances, indices = index.search(query, k=2)

print(f"Indices: {indices}")     # Should be [[0 1]] Apple & Mango closest
print(f"Distances: {distances}") # Small distance = similar