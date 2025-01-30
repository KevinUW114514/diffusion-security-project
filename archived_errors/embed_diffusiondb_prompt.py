import torch, clip, pandas as pd, numpy as np
from sklearn.metrics.pairwise import cosine_similarity

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

diffusiondb = pd.read_csv("sample_head.csv")
diffusiondb = diffusiondb["prompt"].tolist()
diffusiondb = [str(x) for x in diffusiondb]

clip_model, clip_preprocess = clip.load("ViT-L/14", device=DEVICE)
clip_embeddings = []
for i in range(len(diffusiondb) // 500 + 1):
    with torch.no_grad():
        text = clip.tokenize(diffusiondb[i * 500 : (i + 1) * 500], truncate=True).to(DEVICE)
        feature = clip_model.encode_text(text)
        clip_embeddings.append(feature)

# for i in range(len(diffusiondb)):
#     with torch.no_grad():
#         text = clip.tokenize(diffusiondb[i], truncate=True)
#         feature = clip_model.encode_text(text)
#         clip_embeddings.append(feature)

# print(str(x) for x in clip_embeddings[:10])
# for i in range(len(clip_embeddings)):
#     print(f"{clip_embeddings[i].shape}, {len(clip_embeddings)}")
# print("======================================")
# print(f"{clip_embeddings.shape}, {clip_embeddings}")
# for i in range(10):
#     print(f"{clip_embeddings[i]}, {clip_embeddings[i].shape}")

clip_embeddings = torch.cat(clip_embeddings)
# similarity_matrix = cosine_similarity(clip_embeddings)

embeddings_np = clip_embeddings.cpu().numpy()
similarity_matrix = cosine_similarity(embeddings_np)

threshold = 0.95

# Get the indices of all pairs (i, j) where similarity_matrix[i, j] >= threshold
indices_to_remove = np.argwhere(np.triu(similarity_matrix, k=1) >= threshold)[:, 1]

# Get unique indices to remove
to_remove = set(indices_to_remove)

# Use a boolean mask to filter embeddings and prompts
mask = np.ones(len(clip_embeddings), dtype=bool)
mask[list(to_remove)] = False

filtered_embeddings = clip_embeddings[mask]
filtered_prompts = np.array(diffusiondb)[mask]

# Convert back to tensor
filtered_embeddings = torch.stack(filtered_embeddings)

# Save results
torch.save(filtered_embeddings, "filtered_embeddings.pt")

# Save filtered prompts
filtered_df = pd.DataFrame(filtered_prompts, columns=["prompt"])
filtered_df.to_csv("filtered_prompts.csv", index=False)