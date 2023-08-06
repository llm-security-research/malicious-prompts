from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
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
    """
    Cosine similarity between two columns of sentence embeddings

    Args:
      sentence1_emb: sentence1 embedding column
      sentence2_emb: sentence2 embedding column

    Returns:
      The row-wise cosine similarity between the two columns.
      For instance is sentence1_emb=[a,b,c] and sentence2_emb=[x,y,z]
      Then the result is [cosine_similarity(a,x), cosine_similarity(b,y), cosine_similarity(c,z)]
    """
    cos_sim = cosine_similarity(sentence1_emb, sentence2_emb)
    return np.diag(cos_sim)


def log_results(cosine, sentence, sentences, injection_treshold=-1.0):
    header_text = f'Tested prompt: {sentence} - Treshold = {injection_treshold}'
    print(header_text)
    logging.debug(header_text)
    for i, base_prompt in enumerate(sentences):
        match_text = ""
        if cosine[i] >= injection_treshold:
            match_text = "(MATCH)"
            print(f'{match_text}Similarity: {cosine[i]} - {base_prompt}')
        logging.debug(f'{match_text}Similarity: {cosine[i]} - {base_prompt}')
    print()
    logging.debug("")

student_prompts = []

with open('data/tested_prompts.csv', encoding='UTF-8') as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
        student_prompts.append(row[0])

# Load the pre-trained model
model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

# Generate embeddings
start_time = time.time()
student_emb = model.encode(student_prompts, show_progress_bar=True)
logging.debug("--- Encoding time: %s seconds ---" % (time.time() - start_time))

malicious_emb_string = codecs.open('embeddings/malicious_embeddings.json', 'r', encoding='utf-8').read()

prompts_emb = json.loads(malicious_emb_string)
malicious_emb = np.array(list(prompts_emb.values()))
malicious_sentences = np.array(list(prompts_emb.keys()))

start_time = time.time()
for i, value in enumerate(student_emb):
    repeated_student_emb = []
    for j in malicious_emb:
        repeated_student_emb.append(student_emb[i])

    cosine = cos_sim(repeated_student_emb, malicious_emb)

    log_results(cosine, student_prompts[i], malicious_sentences, injection_treshold=0.23)

logging.debug("--- Checking time: %s seconds ---" % (time.time() - start_time))

