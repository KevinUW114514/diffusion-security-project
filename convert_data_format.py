# This file preprocesses the prompt embeddings got previously
# to make it compitable with the ClipCap model.

import torch
import pickle
from tqdm import tqdm

def main():
    path = "filtered_clip_text_normalized_embeddings_checkpoint.pt"
    prompt_name = "filtered_prompts"
    embeddings_name = "filtered_embeddings"
    
    checkpoint_data = torch.load(path)
    pending_prompts = checkpoint_data[prompt_name]
    pending_embeddings = checkpoint_data[embeddings_name]

    all_captions = []
    size = len(pending_prompts)

    for i in tqdm(range(size)):
        d = dict()
        d["clip_embedding"] = i
        d["caption"] = pending_prompts[i]
        all_captions.append(d)
    
    with open("training_data.pkl", 'wb') as f:
        pickle.dump({"clip_embedding":pending_embeddings[: size], "captions": all_captions}, f)

    print('Done')
    # print("%0d embeddings saved " % len(pending_embeddings))
    print("%0d prompts saved " % len(all_captions))
    return 0

if __name__ == '__main__':
    main()
