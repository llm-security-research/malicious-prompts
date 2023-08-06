from sentence_transformers import SentenceTransformer
import numpy as np
import csv
import json

malicious_prompts = []

with open('data/malicious_prompts.csv') as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
        malicious_prompts.append(row[0])

# Load the pre-trained model
model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

# Generate embeddings
malicious_embeddings = model.encode(malicious_prompts, show_progress_bar=True)

malicious_embeddings_list = malicious_embeddings.tolist()

prompts_embeddings = {malicious_prompts[i]: malicious_embeddings_list[i] for i in range(len(malicious_prompts))}

serialized_as_json = json.dumps(prompts_embeddings)
with open('embeddings/malicious_embeddings.json', 'w', encoding="UTF-8") as f:
    f.write(serialized_as_json)