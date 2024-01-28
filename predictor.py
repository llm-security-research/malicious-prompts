import pandas as pd
import numpy as np
import roc_curve
import torch
from sentence_transformers import SentenceTransformer

embeddings = []
categories = []
triplet_embeddings = []
malicious_model = None
model = None
df = None
triplet_df = None


def predict(prompt):

    tested_prompt = model.encode(prompt)
    tested_prompt_predicted = roc_curve.predict(malicious_model, torch.Tensor(tested_prompt))

    least_distance = 999999999999.9
    least_category = "null"

    for index, row in triplet_df.iterrows():
        dist = np.linalg.norm(np.asarray(tested_prompt_predicted) - np.asarray(row['embedding']))
        if least_distance > dist:
            least_distance = dist
            least_category = row['category']

    return least_category, least_distance



if __name__ == "__main__":

    print("Loading siamese dataset...")

    triplet_df = pd.read_csv("data/processed/malignant_siamese.csv", converters={'embedding': pd.eval})
    print(triplet_df.head())

    for index, row in triplet_df.iterrows():
        print(type(row['embedding']))

    print("Loading malicious prompt model...")
    malicious_model = torch.load("./trained/17-01-2024_14-30-03/model.pth")

    print("Loading SentenceTransformers model...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    print("Encoding embeddings...")
    for embedding in embeddings:
        temp = roc_curve.predict(malicious_model, torch.Tensor(embedding))
        triplet_embeddings.append(temp)

    print("Insert prompt to be tested (-1 to exit):")
    while True:
        prompt = input()
        if (prompt == "-1"):
            break
        print(predict(prompt))