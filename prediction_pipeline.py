from sentence_transformers import SentenceTransformer
import torch
import pandas as pd
import ast
import os
import numpy as np
import time
    
class PredictionPipeline:

    def __init__(self, model):
        start = time.time()
        self.m_model = model
        end = time.time()
        print(f'{model} loading time: {end-start}')
        start = time.time()
        self.paraphrase = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=('cuda' if torch.cuda.is_available() else 'cpu'))
        end = time.time()
        print(f'paraphrase-multilingual-MiniLM-L12-v2 loading time: {end-start}')

        def convert_to_list(x):
            return ast.literal_eval(x)

        self.df_malignant = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Malignant/malignant.csv'), converters={'embedding': convert_to_list})

    def predict(self, text):
        start = time.time()
        tested = self.paraphrase.encode(text)
        end = time.time()
        print(f'SentenceTransformer encoding time: {end-start}')

        start = time.time()
        tested_embedding = []
        with torch.no_grad():
            if torch.cuda.is_available():
                tested_embedding = self.m_model.forward_one(torch.Tensor(tested).cuda()).tolist()
            else:
                tested_embedding = self.m_model.forward_one(torch.Tensor(tested)).tolist()
            
        preds = ['none']

        triplet_embeddings = self.df_malignant['embedding']

        least_distance = 999999999999.9
        least_category = "null"
        i = 0
        
        for embedding in triplet_embeddings:
            classified_embedding = []
            with torch.no_grad():
                if torch.cuda.is_available():
                    classified_embedding = self.m_model.forward_one(torch.Tensor(embedding).cuda()).tolist()
                else:
                    classified_embedding = self.m_model.forward_one(torch.Tensor(embedding)).tolist()

            dist = np.linalg.norm(np.asarray(classified_embedding) - np.asarray(tested_embedding))
            if least_distance > dist:
                least_distance = dist
                least_category = self.df_malignant['category'][i]
            i = i+1

            preds[0] = least_category
        
        end = time.time()
        print(f'PromptSentinel prediction time: {end-start}')
        return preds[0]
    
    def get_embedding_from_text(self, text):
        tested = self.paraphrase.encode(text)
        tested_embedding = []
        with torch.no_grad():
            if torch.cuda.is_available():
                tested_embedding = self.m_model.forward_one(torch.Tensor(tested).cuda()).tolist()
            else:
                tested_embedding = self.m_model.forward_one(torch.Tensor(tested)).tolist()

        return tested_embedding

        



