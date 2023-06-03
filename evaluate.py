import random
from pymongo import MongoClient
import torch.optim as optim
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel, GPT2Tokenizer, AlbertTokenizer, AlbertModel, TransfoXLTokenizer, \
    TransfoXLModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, accuracy_score
import torch.nn.functional as F
import seaborn as sns
from pprint import pprint


# Connect to MongoDB
print('Connecting to MongoDB...')
client = MongoClient('mongodb://62.90.89.4:27017/')
db = client['IMDB-data']
print('Successfully connected')

# check if CUDA (GPU) is available and print message
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA (GPU) device: {torch.cuda.get_device_name(device)}")
else:
    device = torch.device("cpu")
    print("CUDA (GPU) not available, using CPU")


def find_best_threshold():
    similar = db['similar_by_IMDB']
    non_similar = db['non_similar_by_IMDB']

    similar_scores = [float(item['cosine - similarity(bert-base-nli-mean-tokens)']) for item in similar.find()]
    non_similar_scores = [float(item['cosine - similarity(bert-base-nli-mean-tokens)']) for item in non_similar.find()]

    labels = [1] * len(similar_scores) + [0] * len(non_similar_scores)
    values = similar_scores + non_similar_scores

    # Iterate over thresholds and calculate accuracy
    thresholds = np.linspace(0, 1, 100)  # Range of thresholds from 0 to 1
    print(thresholds)
    best_threshold = None
    best_accuracy = 0.0

    for threshold in thresholds:
        predicted_labels = [1 if val >= threshold else 0 for val in values]
        accuracy = accuracy_score(labels, predicted_labels)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    # Print the best threshold and accuracy
    print("Best Threshold:", best_threshold)
    print("Best Accuracy:", best_accuracy)


def calculate_scores_and_eval():
    model_name="sentence-transformers/bert-base-nli-mean-tokens"
    movies = db['movies']
    # similar_movies = db['similar_movies']

    y_true = []
    y_pred_cosine = []
    cosine_values = []
    threshold = 0.6

    movies_list = [movie for movie in movies.find()]
    # Loop over all movies in the collection
    for i, movie in enumerate(movies_list):
        # Extract the movie ID and its similar movies from the "more_like_this" list
        movie_id = movie['_id']
        similar_movie_names = movie['more_like_this']
        similar_movies = [m for m in movies.find({'name': {'$in': similar_movie_names}})]
        similar_movie_ids = [m['_id'] for m in similar_movies]

        # Loop over the known similar movies and pair them with the current movie
        embedding1 = torch.tensor(movie[f'{model_name} - embedding'], device=device)
        for similar_movie in similar_movies:
            embedding2 = torch.tensor(similar_movie[f'{model_name} - embedding'], device=device)

            similarity_cosine = torch.nn.functional.cosine_similarity(embedding1.unsqueeze(0),
                                                                      embedding2.unsqueeze(0)).item()

            print(f'similarity_true: 1')
            print(f'\tsimilarity_cosine: {similarity_cosine:.2f}')

            y_true.append(1)
            y_pred_cosine.append(1 if similarity_cosine > threshold else 0)
            cosine_values.append(similarity_cosine)

        # Randomly select the same amount of pairs that are not in the "more_like_this" list
        non_similar_movies = [m for m in movies.find({'_id': {'$ne': movie_id}}) if
                              m['_id'] not in similar_movie_ids and
                              all(non_similar_movie_genre not in movie['genre'] for non_similar_movie_genre in m['genre'])]
        random_non_similar_movies = random.sample(non_similar_movies, len(similar_movie_ids))
        for non_similar_movie in random_non_similar_movies:
            embedding2 = torch.tensor(non_similar_movie[f'{model_name} - embedding'], device=device)

            similarity_cosine = torch.nn.functional.cosine_similarity(embedding1.unsqueeze(0),
                                                                      embedding2.unsqueeze(0)).item()

            print(f'similarity_true: 0')
            print(f'\tsimilarity_cosine: {similarity_cosine:.2f}')

            y_true.append(0)
            y_pred_cosine.append(1 if similarity_cosine > threshold else 0)
            cosine_values.append(similarity_cosine)

        # Print progress every 100 movies
        # if i % 100 == 0:
        print(f"Processed {i + 1} movies.")
        cm_cosine = confusion_matrix(y_true, y_pred_cosine)

        # print('cosine')
        print(cm_cosine)
        print(f'Accuracy: {(((cm_cosine[0][0] + cm_cosine[1][1]) / np.sum(cm_cosine)) * 100):.2f}%\n\n')

    return y_true, cosine_values


def eval_with_calculated_scores():
    similar = db['similar_by_IMDB']
    non_similar = db['non_similar_by_IMDB']

    similar_scores = [float(item['cosine - similarity(bert-base-nli-mean-tokens)']) for item in similar.find()]
    non_similar_scores = [float(item['cosine - similarity(bert-base-nli-mean-tokens)']) for item in non_similar.find()]

    y_true = [1] * len(similar_scores) + [0] * len(non_similar_scores)

    threshold = 0.5
    print(f'Threshold: {threshold}')

    all_scores = similar_scores + non_similar_scores
    y_pred = [1 if item > threshold else 0 for item in all_scores]

    print(f'y_true length = {len(y_true)}')
    print(f'y_pred length = {len(y_pred)}')

    show_results(y_true, y_pred)


def show_results(y_true, y_pred):
    # Get the true labels and predicted labels
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    print(f'Accuracy: {(accuracy_score(y_true, y_pred) * 100):.2f}%\n\n')

    # Plot the confusion matrix
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=['similar', 'non-similar'], yticklabels=['similar', 'non-similar'],
           title='Confusion matrix',
           ylabel='True label',
           xlabel='Predicted label')

    # Add annotations to the confusion matrix
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    # Calculate precision, recall, and F1 score
    print(classification_report(y_true, y_pred))
    plt.show()
