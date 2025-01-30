import torch, pandas as pd
import clip
import copy
import numpy as np

np.random.seed(1919810)

file = pd.read_csv("selected_prompts.csv")
prompts = file["prompt"].to_numpy()
random_indices = np.random.choice(len(prompts), size=2, replace=False)
print(random_indices)
# exit()
num = 9385
print(num)
# random_indices = [9385, 0, 114, 1919, 4237, 8767, 9400]
random_indices = [9386]
prompts = prompts[random_indices]
prompts = copy.deepcopy(prompts)

del file
model, _ = clip.load("ViT-L/14", device="cuda")

calculated_embeedings = torch.load("selected_embeddings.pt")[random_indices]
embeddings = []

promps_embeddings = clip.tokenize(prompts, truncate=True).to("cuda")
with torch.no_grad():
    feature = model.encode_text(promps_embeddings)
    feature = feature / feature.norm(dim=-1, keepdim=True)
    embeddings.append(feature.cpu())
    
embeddings = torch.cat(embeddings)


for i in range(len(prompts)):
    print(f"Prompt: {prompts[i]}")
    print(f"Embedding: {embeddings[i]}")
    print(f"Calculated Embedding: {calculated_embeedings[i]}")
    print(f"Equal: {torch.allclose(embeddings[i], calculated_embeedings[i], atol=1e-2)}")
    print()
