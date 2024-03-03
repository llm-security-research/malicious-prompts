from datetime import datetime
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import os

import torch
import csv
from siamese_network import SiameseNetwork
import csv
import numpy as np
import random

from collections import Counter

import pandas as pd


PATH_CSV = "data/dataset_landmarks_lfw_test.csv"

triplets = []
# set seed for random
# random.seed(7)

# Create an empty dictionary to store the output data
def read_csv(path_csv):
    # csv_data = {}

    # # Open the input CSV file for reading
    # with open(path_csv, "r") as csvfile:
    #     # Create a CSV reader object
    #     reader = csv.reader(csvfile)

    #     # Loop through each row in the CSV file
    #     for row in reader:
    #         # Get the name from the first column of the row
    #         name = row[0]

    #         # If the name is not already in the output_data dictionary, create a new list for the name
    #         if name not in csv_data:
    #             csv_data[name] = []

    #         # Add the row to the list for the current name
    #         csv_data[name].append(list(map(float, row[1:])))

    #     csv_data = {k: v for k, v in csv_data.items() if len(v) > 1}
    #     # print(len(csv_data))
    # return csv_data]
    INPUT_FILE = "data/processed/malignant.csv"
    df = pd.read_csv(INPUT_FILE)
    df.head()

    embeddings = []
    for embedding in df['embedding']:
        temp = [float(x.strip(' []')) for x in embedding.split(',')]
        embeddings.append(temp)

    categories = []
    for category in df['category']:
        categories.append(category)

    print("Vector size:\t\t\t", len(embeddings[0]))
    print("Number of embeddings:\t\t", len(embeddings))
    print("Number of category entries:\t", len(categories))

    data_dict = {}

    for i, embedding in enumerate(embeddings):
        category = categories[i]

        if category not in data_dict:
            data_dict[category] = []

        # data_dict[category].append(list(map(float, embedding)))
        data_dict[category].append(embedding)


    data_dict = {k: v for k, v in data_dict.items() if len(v) > 1}
    return data_dict


def get_item(data_dict, n, reload_triplets):
    global triplets
    if not triplets or reload_triplets:
        for _ in range(n):
            class_anchor = random.choice(list(data_dict.keys()))
            landmarks_anchor = random.choice(data_dict[class_anchor])

            # Select a positive sample from the same class
            landmarks_p = landmarks_anchor
            while landmarks_p == landmarks_anchor:
                landmarks_p = random.choice(data_dict[class_anchor])

            # Select a negative sample from a different class
            class_n = class_anchor

            while class_n == class_anchor:
                class_n = random.choice(list(data_dict.keys()))
            landmarks_n = random.choice(data_dict[class_n])

            # Return the triplets: anchor, positive, negative
            triplets.append((class_anchor, class_n, landmarks_anchor, landmarks_p, landmarks_n))
    return triplets


def predict(model, input_tensor):
    # Make a prediction using the model
    with torch.no_grad():
        if torch.cuda.is_available():
            return model.forward_one(input_tensor.cuda()).tolist()
        else:
            return model.forward_one(input_tensor).tolist()


def euclidean_distance(feature_1, feature_2):
    feature_1 = np.array(feature_1)
    feature_2 = np.array(feature_2)
    dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
    return dist


def evaluate_verification(model, path_csv, n_pairs, reload_triplets=False):
    # Set the model to evaluation mode

    model.cuda()

    model.eval()

    csv_data = read_csv(path_csv)

    triplets = get_item(csv_data, int(n_pairs / 2), reload_triplets=reload_triplets)

    actual = []
    y_scores = []

    for triplet in triplets:
        class_anchor = triplet[0]
        class_n = triplet[1]

        anchor = triplet[2]
        positive = triplet[3]
        negative = triplet[4]

        anchor_encode = predict(model, torch.Tensor(anchor))
        positive_encode = predict(model, torch.Tensor(positive))
        negative_encode = predict(model, torch.Tensor(negative))

        dist_pos = euclidean_distance(anchor_encode, positive_encode)
        dist_neg = euclidean_distance(anchor_encode, negative_encode)

        actual.append(True)
        actual.append(False)

        y_scores.extend([dist_pos, dist_neg])

    y_scores = np.array(y_scores)

    return actual, y_scores


def main(model, path_csv=PATH_CSV, n_pairs=10000, plot=False, folder=None, reload_triplets=False):
    actual, distances = evaluate_verification(model, path_csv, n_pairs, reload_triplets=reload_triplets)

    # Find the maximum distance
    max_distance = np.max(distances)

    # Invert the distances
    distances = max_distance - distances

    # calculate roc curve and plot
    fpr, tpr, thresholds = roc_curve(actual, distances)

    now = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    if not folder:
        date_str = now[:10] 
        folder = f"plots/{date_str}"
        os.makedirs(folder, exist_ok=True)
    with open(f"{folder}/tpr_fpr.txt", "w") as f:
        for i in range(len(fpr)):
            f.write(f'{tpr[i]} {fpr[i]}\n')

    roc_auc = auc(fpr, tpr)

    # Plot best threshold
    # Youden's J statistic
    best_threshold_index = np.argmax(tpr - fpr)
    best_threshold = thresholds[best_threshold_index]

    if plot:
        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.4f)" % roc_auc)
        plt.plot([0, 1], [0, 1], color="navy", linestyle="--", label="Random guess")

        # Plot best threshold point
        best_fpr = fpr[best_threshold_index]
        best_tpr = tpr[best_threshold_index]
        plt.plot(best_fpr, best_tpr, marker="o", color="green", label="Best threshold", lw=0.5)

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC curve - Best threshold: {best_threshold}")
        plt.legend(loc="lower right")

        # Save the plot
        plt.savefig(f"{folder}/roc_curve-{now}.svg", format="svg")
        # plt.show()
        plt.close()
    
    return fpr, tpr, roc_auc, best_threshold

def get_model_size(model):
        return sum(p.numel() for p in model.parameters())
        
if __name__ == "__main__":
    # Initialize the model
    model = SiameseNetwork(128)  # replace with your actual model class

    # Load the state dictionary from the .pth file
    state_dict = torch.load("best_model/state.pth")

    # Load the state dictionary into the model
    model.load_state_dict(state_dict)

    # print(f'Model size: {get_model_size(model)} parameters')

    for _ in range(10):
        fpr, tpr, roc_auc, best_threshold = main(model, plot=True, reload_triplets=True)
        print(f'ROC AUC: {roc_auc}')
        # print(f'Best threshold: {best_threshold}')