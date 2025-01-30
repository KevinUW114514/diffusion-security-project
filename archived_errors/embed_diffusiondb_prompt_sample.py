import torch, clip, pandas as pd, numpy as np

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cuda"

# diffusiondb = pd.read_csv("sample_head.csv")
diffusiondb = pd.read_csv("diffusiondb.csv")
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

# Filter out the rows that need to be removed
filtered_vectors = clip_embeddings[~to_remove]

filtered_prompts = [diffusiondb[i] for i in range(len(diffusiondb)) if not to_remove[i].item()]

# Save filtered embeddings to a file (e.g., .pt for PyTorch or .npy for NumPy)
torch.save(filtered_vectors, "filtered_embeddings.pt")  # Save as a PyTorch tensor
np.save("filtered_embeddings.npy", filtered_vectors.cpu().numpy())

# Save results
filtered_df = pd.DataFrame(filtered_prompts, columns=["prompt"])
filtered_df.to_csv("filtered_prompt.csv", index=False)