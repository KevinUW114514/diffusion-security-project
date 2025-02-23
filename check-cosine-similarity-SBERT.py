from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")
sentence1 = "A dog running in a field"
sentence2 = "A horse sprinting through grass"

emb1 = model.encode(sentence1, convert_to_tensor=True)
emb2 = model.encode(sentence2, convert_to_tensor=True)

cosine_sim = util.pytorch_cos_sim(emb1, emb2)
print(f"SBERT Cosine Similarity: {cosine_sim.item():.4f}")
