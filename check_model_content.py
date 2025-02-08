import torch

state_dict = torch.load("./checkpoints/coco_prefix-009.pt", map_location=torch.device('cpu'))

# Create a new state_dict with remapped keys
# new_state_dict = {}
# for key, value in state_dict.items():
#     if key.startswith("clip_project.model"):
#         # Map "clip_project.model.X" to "clip_project.X"
#         new_key = key.replace("clip_project.model", "clip_project")
#         new_state_dict[new_key] = value
#     else:
#         new_state_dict[key] = value
for k in state_dict.keys():
    print(k)

# Load the modified state_dict
# model.load_state_dict(new_state_dict)
