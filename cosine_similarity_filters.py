import torch
from tqdm import tqdm
import pickle

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

with open("./data/clip_text_normalized_embeddings_checkpoint.pkl", "rb") as f:
    checkpoint_data = pickle.load(f)
    
embeddings = checkpoint_data["embeddings"]
prompts = checkpoint_data["prompts"]

def filter_embeddings_gpu(embeddings, batch_size=1000, threshold=0.9):
    embeddings = embeddings.to(device)

    # Initialize result set with the first embedding
    result_set = embeddings[0].unsqueeze(0)  # Ensure it's 2D
    result_indices = [0]  # Track indices of added vectors

    num_embeddings = embeddings.shape[0]

    for i in tqdm(range(1, num_embeddings, batch_size), desc="Processing embeddings"):
        # Get the current batch and move to GPU
        batch_vectors = embeddings[i:i + batch_size]

        # Compute cosine similarity: batch_vectors @ result_set.T
        cosine_sims = batch_vectors @ result_set.T

        # Identify vectors whose similarities are all below the threshold
        mask = (cosine_sims < threshold).all(dim=1)
        
        actual_batch_size = batch_vectors.shape[0]  # Handle final batch size properly
        filtered_indices = torch.arange(i, i + actual_batch_size, device=device)[mask]

        # Filter the vectors and indices that meet the condition
        filtered_vectors = batch_vectors[mask]

        # Append to result set and track indices
        if filtered_vectors.size(0) > 0:
            result_set = torch.cat([result_set, filtered_vectors], dim=0)
            result_indices.extend(filtered_indices.cpu().tolist())  # Move indices to CPU for storage
        
        # if len(result_indices) >= 10:
        #     return result_set.cpu(), result_indices

    return result_set.cpu(), result_indices  # Move result set to CPU for further use

result_set, result_indices = filter_embeddings_gpu(embeddings)
prompts = prompts[result_indices]  # Filter prompts based on selected indices

with open("./data/filtered_clip_text_normalized_embeddings_checkpoint.pkl", "wb") as f:
    pickle.dump({
        "filtered_embeddings": result_set,
        "filtered_prompts": prompts,
        "filtered_indices": result_indices
    }, f)

print(f"Number of filtered embeddings: {result_set.shape[0]}")
print(f"Selected indices: {result_indices[:10]}")  # Show the first 10 indices

