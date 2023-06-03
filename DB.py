import json
import random
from pymongo import MongoClient
import torch


print('Connecting to MongoDB...')
client = MongoClient('mongodb://62.90.89.4:27017/')
db = client['IMDB-data']
movie = db['movies']
print('Successfully connected')

# check if CUDA (GPU) is available and print message
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA (GPU) device: {torch.cuda.get_device_name(device)}")
else:
    device = torch.device("cpu")
    print("CUDA (GPU) not available, using CPU")


def create_DB():
    with open("data.json", "r") as file:
        data = json.load(file)

        for mov in data:
            data[mov]["actor"] = [item["name"] for item in data[mov]["actor"]]
            data[mov]["director"] = [item["name"] for item in data[mov]["director"]]
            data[mov]["creator"] = [item["name"] for item in data[mov]["creator"]]
            movie.insert_one(data[mov])


def remove_duplicates_and_empty_lists():
    """
    Removes duplicates and empty 'more_like_this' lists from movie documents.

    Iterates over movie documents and checks for duplicate entries or empty 'more_like_this' likes.
    If duplicates or empty likes are found, they are either removed or updated to ensure uniqueness and validity.
    """
    movies = [m for m in movie.find()]
    flag = True

    for index2, mov in enumerate(movies):
        likes = mov['more_like_this']
        new_likes = []

        for index, like in enumerate(likes):
            try:
                if movie.find_one({"name": like}):
                    new_likes.append(like)
            except:
                continue

        if index2 % 1000 == 0:
            print(index2)

        if len(likes) == len(list(set(new_likes))):
            continue
        if len(new_likes) == 0:
            movie.delete_one({"_id": mov["_id"]})
            print(f"remove {mov['name']}")
            flag = False
        else:
            new_likes = list(set(new_likes))
            print(movie.update_one({"_id": mov["_id"]}, {"$set": {'more_like_this': new_likes}}))
            flag = False
    return flag


def create_similar_pairs_collection():
    movies = db['movies']

    # Create a new collection to store similar movie pairs
    similar_movies = db['similar_by_IMDB']
    x = 1
    # Loop over each movie in the collection
    movies_list = [item for item in movies.find()]
    for movie in movies_list:
        # Loop over the movie names in the movie's more_like_this list
        for similar_movie_name in movie['more_like_this']:
            # Find the corresponding movie document in the collection using its name
            similar_movie = movies.find_one({'name': similar_movie_name})
            # If the movie document exists and it's not the same as the current movie
            if similar_movie and similar_movie['_id'] != movie['_id']:
                # Add a new document to the similar_movies collection
                embedded_1 = torch.tensor(movies.find_one({"_id": movie['_id']})["sentence-transformers/bert-base-nli-mean-tokens - embedding"], device=device)
                embedded_2 = torch.tensor(movies.find_one({"_id": similar_movie['_id']})["sentence-transformers/bert-base-nli-mean-tokens - embedding"], device=device)
                similarity_cosine = torch.nn.functional.cosine_similarity(embedded_1.unsqueeze(0),
                                                                          embedded_2.unsqueeze(0)).item()
                similar_movies.insert_one({
                    'pair': x,
                    'movie1': movie['_id'],
                    'movie2': similar_movie['_id'],
                    "cosine - similarity(bert-base-nli-mean-tokens)": round(similarity_cosine, 4)
                })
                if x % 1000 == 0:
                    print(x)
                x += 1


def create_non_similar_pairs_collection():
    movies = db['movies']

    # Create a new collection to store similar movie pairs
    non_similar_movies_collection = db['non_similar_by_IMDB']

    # Loop over each movie in the collection
    movies_list = [item for item in movies.find()]
    for i, movie in enumerate(movies_list):
        # Randomly select the same amount of pairs that are not in the "more_like_this" list
        non_similar_movies = [m for m in movies.find({'_id': {'$ne': movie['_id']}}) if
                              m['name'] not in movie['more_like_this'] and
                              all(non_similar_movie_genre not in movie['genre'] for non_similar_movie_genre in m['genre'])]
        random_non_similar_movies = random.sample(non_similar_movies, len(movie['more_like_this']) * 2)

        candidates = []

        # Loop over the movie names in the movie's more_like_this list
        for non_similar_movie in random_non_similar_movies:
            # Add a new document to the non_similar_movies collection
            embedded_1 = torch.tensor(movie["sentence-transformers/bert-base-nli-mean-tokens - embedding"], device=device)
            embedded_2 = torch.tensor(non_similar_movie["sentence-transformers/bert-base-nli-mean-tokens - embedding"], device=device)
            similarity_cosine = torch.nn.functional.cosine_similarity(embedded_1.unsqueeze(0), embedded_2.unsqueeze(0)).item()

            candidates.append({
                'movie1': movie['_id'],
                'movie2': non_similar_movie['_id'],
                "cosine - similarity(bert-base-nli-mean-tokens)": round(similarity_cosine, 4)
            })

        # Sort the candidates based on similarity in ascending order
        sorted_candidates = sorted(candidates, key=lambda x: x['cosine - similarity(bert-base-nli-mean-tokens)'])

        # Get the 4 lowest similarity values
        lowest_similarities = sorted_candidates[:(len(candidates) // 2)]

        [non_similar_movies_collection.insert_one(candidate) for candidate in lowest_similarities]

        if i % 1000 == 0:
            print(i)
        i += 1


def fix_html_encoding():
    # &apos; = '
    movies = db['movies']

    all_movies = [m for m in movies.find()]
    for i, movie in enumerate(all_movies):
        name = movie.get('name')
        if name:
            # Replace "&apos;" with "'"
            name = name.replace("&apos;", "'")
            movie['name'] = name
            # movies.replace_one({'_id': movie['_id']}, movie)

        desc = movie.get('description')
        if desc:
            # Replace "&apos;" with "'"
            desc = desc.replace("&apos;", "'")
            movie['description'] = desc

        movies.replace_one({'_id': movie['_id']}, movie)

        if i % 1000 == 0:
            print(f"Processed {i + 1} movies.")


def delete_duplicates():
    similar_movies = db['similar_movies']
    movies = db['movies']
    print(similar_movies.count_documents({'cosine-similarity (bert-base-nli-mean-tokens)': 1}))
    docs1 = similar_movies.find({'cosine-similarity (bert-base-nli-mean-tokens)': 1})
    # [print(f"{doc['movie1']}\t{doc['movie2']}") for doc in docs1]
    c = 0
    for doc in docs1:
        mov1 = movies.find_one({'_id': doc['movie1']})
        mov2 = movies.find_one({'_id': doc['movie2']})

        if not mov1 or not mov2:
            similar_movies.delete_one({'movie1': doc['movie1'], 'movie2': doc['movie2']})
            continue

        if mov1['name'] == mov2['name'] and mov1['description'] == mov2['description']:
            # print(f"{mov1['name']}\t{mov2['name']}")
            c += 1
            movies.delete_one({'_id': doc['movie2']})
            similar_movies.delete_one({'movie1': doc['movie1'], 'movie2': doc['movie2']})

    print(similar_movies.count_documents({'cosine-similarity (bert-base-nli-mean-tokens)': 1}))
    print(c)
    return


def update_cosine_scores_for_all_documents():
    similar_movies = db['non_similar_movies']
    movies = db['movies']
    s = [mov for mov in similar_movies.find()]
    for index2, similar in enumerate(s):
        embedded_1 = torch.tensor(movies.find_one({"_id": similar["movie1"]})["sentence-transformers/bert-base-nli-mean-tokens - embedding"], device=device)
        embedded_2 = torch.tensor(movies.find_one({"_id": similar["movie2"]})["gpt2-embedding"], device=device)
        similarity_cosine = torch.nn.functional.cosine_similarity(embedded_1.unsqueeze(0),
                                                                  embedded_2.unsqueeze(0)).item()
        similar_movies.update_one({"_id": similar["_id"]}, {"$set": {'cosine-similarity\
         (gpt2)': round(similarity_cosine, 4)}})
        print(f"run number {index2 + 1}")
        print(f'\tsimilarity_cosine (gpt2): {similarity_cosine:.2f}')


        embedded_1 = torch.tensor(movies.find_one({"_id": similar["movie1"]})["bert-base-nli-mean-tokens-embedding"],
                                  device=device)
        embedded_2 = torch.tensor(movies.find_one({"_id": similar["movie2"]})["bert-base-nli-mean-tokens-embedding"],
                                  device=device)
        similarity_cosine = torch.nn.functional.cosine_similarity(embedded_1.unsqueeze(0),
                                                                  embedded_2.unsqueeze(0)).item()
        similar_movies.update_one({"_id": similar["_id"]}, {
            "$set": {'cosine-similarity(bert-base-nli-mean-tokens)': round(similarity_cosine, 4)}})
        print(f'\tsimilarity_cosine (bert-base-nli-mean-tokens): {similarity_cosine:.2f}')

        embedded_1 = torch.tensor(movies.find_one({"_id": similar["movie1"]})["roberta-base-embedding"], device=device)
        embedded_2 = torch.tensor(movies.find_one({"_id": similar["movie2"]})["roberta-base-embedding"], device=device)
        similarity_cosine = torch.nn.functional.cosine_similarity(embedded_1.unsqueeze(0),
                                                                  embedded_2.unsqueeze(0)).item()
        similar_movies.update_one({"_id": similar["_id"]},
                                  {"$set": {'cosine-similarity(roberta-base)': round(similarity_cosine, 4)}})
        print(f'\tsimilarity_cosine (roberta-base): {similarity_cosine:.2f}')

        embedded_1 = torch.tensor(movies.find_one({"_id": similar["movie1"]})["xlnet-base-cased-embedding"],
                                  device=device)
        embedded_2 = torch.tensor(movies.find_one({"_id": similar["movie2"]})["xlnet-base-cased-embedding"],
                                  device=device)
        similarity_cosine = torch.nn.functional.cosine_similarity(embedded_1.unsqueeze(0),
                                                                  embedded_2.unsqueeze(0)).item()
        similar_movies.update_one({"_id": similar["_id"]},
                                  {"$set": {'cosine-similarity(xlnet-base-cased)': round(similarity_cosine, 4)}})
        print(f'\tsimilarity_cosine (xlnet-base-cased): {similarity_cosine:.2f}')

        embedded_1 = torch.tensor(movies.find_one({"_id": similar["movie1"]})["bert-large-uncased - embedding"],
                                  device=device)
        embedded_2 = torch.tensor(movies.find_one({"_id": similar["movie2"]})["bert-large-uncased - embedding"],
                                  device=device)
        similarity_cosine = torch.nn.functional.cosine_similarity(embedded_1.unsqueeze(0),
                                                                  embedded_2.unsqueeze(0)).item()
        similar_movies.update_one({"_id": similar["_id"]},
                                  {"$set": {'cosine-similarity(bert-large-uncased)': round(similarity_cosine, 4)}})
        print(f'\tsimilarity_cosine (bert-large-uncased): {similarity_cosine:.2f}')