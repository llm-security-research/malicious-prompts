from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import csv
import json
import codecs
import time
from datetime import datetime
import logging

model = SentenceTransformer('distiluse-base-multilingual-cased-v1')