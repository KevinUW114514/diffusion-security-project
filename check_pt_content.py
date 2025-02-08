import torch

# Replace 'my_model.pt' with the path to your .pt file
checkpoint = torch.load('../diffusion/result.pt', map_location=torch.device('cpu'))

# If this is a state_dict or a dictionary with multiple items, you can inspect its keys
print(checkpoint)
