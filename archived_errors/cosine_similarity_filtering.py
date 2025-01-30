import torch

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Example tensor (millions of entries)
tensor = torch.arange(10_000_000, device=device)  # Replace with your actual tensor

# Calculate the number of elements to select
num_elements = tensor.numel()
num_to_select = num_elements // 4

# Generate random indices on the GPU
indices = torch.randperm(num_elements, device=device)[:num_to_select]

# Select the elements using the indices
selected_elements = tensor[indices]

# If you need the original indices (positions) in the tensor
original_indices = indices

print("Selected elements:", selected_elements)
print("Original indices:", original_indices)