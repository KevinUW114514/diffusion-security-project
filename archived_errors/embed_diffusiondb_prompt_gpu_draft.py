import torch, clip, pandas as pd, numpy as np
import os

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cuda"
checkpoint_path = "clip_embeddings_checkpoint.pt"
assert torch.cuda.device_count() >= 2, "At least 2 GPUs are required"
device_0 = torch.device("cuda:0")
device_1 = torch.device("cuda:1")

if os.path.exists(checkpoint_path):
    print("Loading checkpoint...")
    checkpoint = torch.load("clip_embeddings_checkpoint.pt")
    clip_embeddings = checkpoint["clip_embeddings"].to(DEVICE)  # Move embeddings to GPU
    norm_vectors = checkpoint["norm_vectors"].to(DEVICE)        # Move normalized embeddings to GPU
    diffusiondb = checkpoint["diffusiondb"]
else:
    diffusiondb = pd.read_csv("diffusiondb.csv")
    # diffusiondb = pd.read_csv("sample_head.csv")
    diffusiondb = diffusiondb["prompt"].tolist()
    diffusiondb = [str(x) for x in diffusiondb]

    clip_model, clip_preprocess = clip.load("ViT-L/14", device=DEVICE)
    clip_embeddings = []
    for i in range(len(diffusiondb) // 500 + 1):
        with torch.no_grad():
            text = clip.tokenize(diffusiondb[i * 500 : (i + 1) * 500], truncate=True).to(DEVICE)
            feature = clip_model.encode_text(text)
            clip_embeddings.append(feature)

    clip_embeddings = torch.cat(clip_embeddings)
    # similarity_matrix = cosine_similarity(clip_embeddings)

    # Normalize the vectors to unit length (to compute cosine similarity)
    norm_vectors = clip_embeddings / clip_embeddings.norm(dim=1, keepdim=True)

    torch.save({
        "clip_embeddings": clip_embeddings.cpu(),  # Save embeddings (move to CPU for storage efficiency)
        "norm_vectors": norm_vectors.cpu(),        # Save normalized embeddings
        "diffusiondb": diffusiondb                 # Save original text prompts
    }, "clip_embeddings_checkpoint.pt")

    print("Checkpoint saved to clip_embeddings_checkpoint.pt")
    


"""
# Compute cosine similarity using vectorization
cosine_similarity_matrix = torch.mm(norm_vectors, norm_vectors.T)

# Create a mask to ignore self-similarity (diagonal)
mask = torch.eye(cosine_similarity_matrix.size(0), device=DEVICE).bool()
cosine_similarity_matrix.masked_fill_(mask, 0)

# Find pairs where cosine similarity > 0.95
threshold = 0.95
similar_pairs = cosine_similarity_matrix > threshold

# Identify rows to remove (keep the first occurrence of each similar pair)
to_remove = torch.any(similar_pairs.triu(1), dim=0)
"""

threshold = 0.95
num_vectors = norm_vectors.size(0)
batch_size = 5000
to_remove = torch.zeros(num_vectors, dtype=torch.bool, device=DEVICE)  # Boolean mask

for i in range(0, num_vectors, batch_size):
    chunk = norm_vectors[i:i + batch_size]  # Current chunk
    similarity_chunk = torch.mm(chunk, norm_vectors.T)  # Compute similarity

    # Ignore self-similarity
    diagonal_indices = torch.arange(i, min(i + batch_size, num_vectors), device=DEVICE)
    similarity_chunk[:, diagonal_indices] = 0

    # Find pairs exceeding the threshold
    mask = similarity_chunk > threshold

    # Update removal mask
    to_remove[i:i + batch_size] = mask.any(dim=1)  # Mark rows for removal if they match any row

# Filter out the rows that need to be removed
filtered_vectors = clip_embeddings[~(to_remove.cpu())]

filtered_prompts = [diffusiondb[i] for i in range(len(diffusiondb)) if not to_remove[i].item()]

# Save filtered embeddings to a file (e.g., .pt for PyTorch or .npy for NumPy)
torch.save(filtered_vectors.cpu(), "filtered_embeddings.pt")  # Save as a PyTorch tensor
np.save("filtered_embeddings.npy", filtered_vectors.cpu().numpy())

# Save results
filtered_df = pd.DataFrame(filtered_prompts, columns=["prompt"])
filtered_df.to_csv("filtered_prompt.csv", index=False)