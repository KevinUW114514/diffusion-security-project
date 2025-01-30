import torch
import pandas as pd, numpy as np

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

random_seed = 114514
torch.manual_seed(random_seed)
if device.type == 'cuda':
    torch.cuda.manual_seed_all(random_seed)

# Load the prompts from the CSV file
diffusiondb = pd.read_csv("diffusiondb.csv")
prompts = diffusiondb["prompt"].to_numpy()
prompts = np.array([str(x) for x in prompts]) 

# Load the embeddings
embeddings = torch.load("clip_text_normalized_embeddings_checkpoint1.pt")["normalized_embeddings"].to(device)

# Ensure embeddings is a tensor
if not isinstance(embeddings, torch.Tensor):
    embeddings = torch.tensor(embeddings)

# Calculate the number of elements to select
num_elements = embeddings.shape[0]  # Assuming embeddings is a 2D tensor (num_prompts, embedding_dim)
num_to_select = num_elements // 4

# Generate random indices on the GPU
indices = torch.randperm(num_elements, device=device)[:num_to_select]

# Select the elements using the indices
selected_embeddings = embeddings[indices].cpu()
selected_prompts = prompts[indices.cpu().numpy()]

# Save the selected prompts and embeddings to new files
# Save selected prompts to a new CSV file
selected_prompts_df = pd.DataFrame(selected_prompts, columns=["prompt"])
selected_prompts_df.to_csv("selected_prompts.csv", index=False)

# Save selected embeddings to a new PT file
torch.save(selected_embeddings, "selected_embeddings.pt")

# Save the selected indices to another file
torch.save(indices.cpu(), "selected_indices.pt")

print("Selected prompts, embeddings, and indices have been saved.")
print("Number of selected prompts:", len(selected_prompts))
print("Number of indice:", len(indices))