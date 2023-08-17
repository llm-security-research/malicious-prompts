from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import csv
import json
import codecs
import time
from datetime import datetime
import logging

now = datetime.now()
current_time = now.strftime("%d-%m-%Y %H-%M-%S")
print(current_time)
log_file = f'logs/log {current_time}.txt'
logging.basicConfig(filename=log_file, level=logging.DEBUG, encoding='utf-8')

def cos_sim(sentence1_emb, sentence2_emb):
    cos_sim = cosine_similarity(sentence1_emb, sentence2_emb)
    return np.diag(cos_sim)

def get_is_malicious(cosine, malicious_treshold=0.25):
    malicious = False

    for cosine in cosines:
        if cosine >= malicious_treshold:
            malicious = True
            break
    
    return malicious
            
    
def log_results(cosine, tested_sentence, sentences, malicious_treshold=-1.0):
    header_text = f'Tested prompt: {tested_sentence} - Treshold = {malicious_treshold}'
    print(header_text)
    logging.debug(header_text)
    for i, base_prompt in enumerate(sentences):
        match_text = ""
        if cosine[i] >= malicious_treshold:
            match_text = "(MATCH)"
            print(f'{match_text}Similarity: {cosine[i]} - {base_prompt}')
        logging.debug(f'{match_text}Similarity: {cosine[i]} - {base_prompt}')
    print()
    logging.debug("")

def find_best_treshold(iterations):
    treshold = 0.0
    best_results = 0
    i = 0
    while i < iterations:
        
        i = i + 1

    return treshold

tested_prompts = pd.read_csv('data/no_malicious_prompts.csv', names=['prompt', 'malicious'], header=None)

tested_prompts_list = tested_prompts['prompt'].tolist()

# with open('data/tested_prompts.csv', encoding='UTF-8') as csvfile:
#     spamreader = csv.reader(csvfile)
#     for row in spamreader:
#         tested_prompts.append(row[0])

# Load the pre-trained model
model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

# Generate embeddings
start_time = time.time()
tested_emb = model.encode(tested_prompts_list, show_progress_bar=True)
logging.debug("--- Encoding time: %s seconds ---" % (time.time() - start_time))

malicious_emb_string = codecs.open('embeddings/malicious_embeddings.json', 'r', encoding='utf-8').read()

prompts_emb = json.loads(malicious_emb_string)
malicious_emb = np.array(list(prompts_emb.values()))
malicious_sentences = np.array(list(prompts_emb.keys()))

start_time = time.time()
for i, value in enumerate(tested_emb):
    repeated_tested_emb = []
    for j in malicious_emb:
        repeated_tested_emb.append(tested_emb[i])

    cosines = cos_sim(repeated_tested_emb, malicious_emb)

    malicious = get_is_malicious(cosines, malicious_treshold=0.25)
    if malicious == True:
        print(tested_prompts_list[i], malicious)
    # log_results(cosines, tested_prompts_list[i], malicious_sentences, malicious_treshold=0.25)

logging.debug("--- Checking time: %s seconds ---" % (time.time() - start_time))

