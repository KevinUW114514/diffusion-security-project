import pickle

with open("./data/filtered_clip_text_normalized_embeddings_checkpoint.pkl", "rb") as f:
    checkpoint_data = pickle.load(f)
    
pending_prompts = checkpoint_data["filtered_prompts"].reset_index(drop=True)
pending_embeddings = checkpoint_data["filtered_embeddings"]

print(f"Shape: {pending_embeddings.shape}")

print(f"Number of pending prompts: {len(pending_prompts)}")
print(f"Number of pending embeddings: {pending_embeddings.shape[0]}")
