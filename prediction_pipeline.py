from sentence_transformers import SentenceTransformer
import torch
import pandas as pd
import ast
import os
import numpy as np
    
class PredictionPipeline:

    def __init__(self, modelaa):
        self.m_model = modelaa

        def convert_to_list(x):
            return ast.literal_eval(x)

        self.df_malignant = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Malignant/malignant.csv'), converters={'embedding': convert_to_list})

    def predict(self, text):

        paraphrase = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        tested = paraphrase.encode(text)
        tested_embedding = []
        with torch.no_grad():
            tested_embedding = self.m_model.forward_one(torch.Tensor(tested).cuda()).tolist()
            
        preds = ['none']

        triplet_embeddings = self.df_malignant['embedding']

        least_distance = 999999999999.9
        least_category = "null"
        i = 0
        
        for embedding in triplet_embeddings:
            classified_embedding = []
            with torch.no_grad():
                classified_embedding = self.m_model.forward_one(torch.Tensor(embedding).cuda()).tolist()

            dist = np.linalg.norm(np.asarray(classified_embedding) - np.asarray(tested_embedding))
            if least_distance > dist:
                least_distance = dist
                least_category = self.df_malignant['category'][i]
            i = i+1

            preds[0] = least_category
        
        return preds[0]


