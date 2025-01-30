import torch, pandas as pd
import clip
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

random_seed = 1919810
# torch.manual_seed(random_seed)

# np.random.seed(random_seed)
# if device.type == 'cuda':
#     torch.cuda.manual_seed_all(random_seed)

model, _ = clip.load("ViT-L/14", device="cuda")

path = "filtered_clip_text_normalized_embeddings_checkpoint.pt"
check_point_data = torch.load(path)
prompts = check_point_data["filtered_prompts"]
calculated_embeedings = check_point_data["filtered_embeddings"]
embeddings = []

assert prompts.shape[0] == calculated_embeedings.shape[0], "wtf?"
indices = np.random.choice(prompts.size, size=5000, replace=False)

prompts = prompts[indices]
calculated_embeedings = calculated_embeedings[indices]

promps_embeddings = clip.tokenize(prompts, truncate=True).to("cuda")
with torch.no_grad():
    feature = model.encode_text(promps_embeddings)
    feature = feature / feature.norm(dim=-1, keepdim=True)
    embeddings.append(feature.cpu())
    
embeddings = torch.cat(embeddings)

# for i in range(len(prompts)):
#     # print(f"Prompt: {prompts[i]}")
#     # print(f"Embedding: {embeddings[i]}")
#     # print(f"Calculated Embedding: {calculated_embeedings[i]}")
#     print(f"Equal: {torch.allclose(embeddings[i], calculated_embeedings[i], atol=1e-2)}")
#     print()
print(f"Prompt: {prompts[0]}")
print(f"Equal: {torch.allclose(embeddings, calculated_embeedings, atol=1e-2)}")
