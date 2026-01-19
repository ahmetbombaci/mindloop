from sentence_transformers import SentenceTransformer

model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

texts = [
    "How do I reset my VPN password?",
    "Change VPN credentials"
]

vectors = model.encode(texts)
print(len(vectors), len(vectors[0]))
