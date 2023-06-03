import matplotlib.pyplot as plt
from collections import Counter
# from wordcloud import WordCloud
import datetime
from pymongo import MongoClient
import numpy as np
from sklearn.metrics import accuracy_score
import seaborn as sns

# Connect to MongoDB
print('Connecting to MongoDB...')
client = MongoClient('mongodb://62.90.89.4:27017/')
db = client['IMDB-data']
movie_database = [item for item in db['mini_movies_db'].find()]
print('Successfully connected')


def plot_cumulative():
    similar = db['similar_by_IMDB']
    non_similar = db['non_similar_by_IMDB']

    similar_scores = [float(item['cosine - similarity(bert-base-nli-mean-tokens)']) for item in similar.find()]
    non_similar_scores = [float(item['cosine - similarity(bert-base-nli-mean-tokens)']) for item in non_similar.find()]

    # Calculate cumulative distribution
    sorted_values = np.sort(similar_scores)
    cumulative = np.arange(len(sorted_values)) / float(len(sorted_values))

    # Plotting the cumulative distribution
    plt.subplot(211)
    plt.plot(sorted_values, cumulative, marker='o')
    plt.xlabel('Values')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Distribution of Values for Label 1')

    # Calculate cumulative distribution
    sorted_values = np.sort(non_similar_scores)
    cumulative = np.arange(len(sorted_values)) / float(len(sorted_values))

    # Set the x-axis limits for subplot 1
    xlim = plt.xlim()

    # Plotting the cumulative distribution
    plt.subplot(212)
    plt.plot(sorted_values, cumulative, marker='o')
    plt.xlabel('Values')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Distribution of Values for Label 0')

    # Set the x-axis limits for subplot 2 to match subplot 1
    plt.xlim(xlim)

    plt.tight_layout()
    plt.show()


def plot_distribution():
    similar = db['similar_by_IMDB']
    non_similar = db['non_similar_by_IMDB']

    similar_scores = [float(item['cosine - similarity(bert-base-nli-mean-tokens)']) for item in similar.find()]
    non_similar_scores = [float(item['cosine - similarity(bert-base-nli-mean-tokens)']) for item in non_similar.find()]

    # Smoothing using kernel density estimation (KDE)
    kde_similar = sns.kdeplot(similar_scores, fill=True)
    kde_non_similar = sns.kdeplot(non_similar_scores, fill=True)

    # Set labels and titles
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.title('Smoothed Distribution of Values')

    # Add legend
    plt.legend(['Label 1', 'Label 0'])

    plt.show()
    return

    # # Plotting the distribution
    # plt.subplot(211)
    # plt.hist(similar_scores, bins=1000)
    # plt.xlabel('Values')
    # plt.ylabel('Frequency')
    # plt.title('Distribution of Values for Label 1')
    #
    # # Set the x-axis limits for subplot 1
    # xlim = plt.xlim()
    #
    # # Plotting the distribution
    # plt.subplot(212)
    # plt.hist(non_similar_scores, bins=1000)
    # plt.xlabel('Values')
    # plt.ylabel('Frequency')
    # plt.title('Distribution of Values for Label 0')
    #
    # # Set the x-axis limits for subplot 2 to match subplot 1
    # plt.xlim(xlim)
    #
    # plt.tight_layout()
    # plt.show()


def calculate_statistics_for_cosine_scores():
    similar_movies = db['similar_movies']
    movies = db['movies']
    docs = similar_movies.find()
    cos_values = [doc['cosine-similarity (bert-base-nli-mean-tokens)'] for doc in docs]

    counter = 0
    for item in cos_values:
        if item >= 0.5:
            counter += 1
    print(f"Over 0.5 percent: {(counter/len(cos_values)):.2f}%")

    min_cos = np.min(cos_values)
    max_cos = np.max(cos_values)
    mean_cos = np.mean(cos_values)
    median_cos = np.median(cos_values)
    std_cos = np.std(cos_values)

    print(f'Min: {min_cos}')
    print(f'Max: {max_cos}')
    print(f'Mean: {mean_cos}')
    print(f'Median: {median_cos}')
    print(f'Std: {std_cos}')

    sns.histplot(cos_values, kde=True)
    plt.title('Histogram of similar movies')
    plt.xlabel('cos-similar')
    plt.show()


# Total Number of Movies
total_movies = len(movie_database)
print("Total number of movies:", total_movies)

# Average Rating
total_ratings = sum(movie['rating']['ratingValue'] for movie in movie_database if movie['rating']['ratingValue'])
average_rating = total_ratings / total_movies
print("Average rating:", average_rating)

# Most Common Genres
genres = [genre for movie in movie_database if movie['genre'] for genre in movie['genre']]
common_genres = Counter(genres).most_common(5)
print("Most common genres:")
for genre, count in common_genres:
    print(genre, "-", count)

# Movies by Year
movies_by_year = {}
for movie in movie_database:
    if not movie['datePublished']:
        continue
    year = movie['datePublished'].split('-')[0]
    if year in movies_by_year:
        movies_by_year[year] += 1
    else:
        movies_by_year[year] = 1

print("Movies by year:")
for year, count in sorted(movies_by_year.items()):
    print(year, ":", count)


# Longest and Shortest Movies
def duration_to_minutes(duration):
    parts = duration.split('T')[1].split('H')
    hours = int(parts[0]) if parts[0] else 0
    minutes = int(parts[1].rstrip('M')) if parts[1].rstrip('M') else 0
    return hours * 60 + minutes


movie_durations = [duration_to_minutes(movie['duration']) for movie in movie_database]
longest_movie = max(movie_durations)
shortest_movie = min(movie_durations)

print("Longest movie duration:", str(datetime.timedelta(minutes=longest_movie)))
print("Shortest movie duration:", str(datetime.timedelta(minutes=shortest_movie)))

# Most Common Keywords
all_keywords = [keyword.strip() for movie in movie_database if 'keywords' in movie.keys() and movie['keywords'] for
                keyword in movie['keywords'].split(',')]
common_keywords = Counter(all_keywords).most_common(5)
print("Most common keywords:")
for keyword, count in common_keywords:
    print(keyword, "-", count)

# Distribution of Ratings
ratings = [movie['rating']['ratingValue'] for movie in movie_database if movie['rating']['ratingValue']]
plt.hist(ratings, bins=10)
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.title('Distribution of Ratings')
plt.show()

# Top Actors by Movie Count
all_actors = [actor for movie in movie_database for actor in movie['actor']]
top_actors = Counter(all_actors).most_common(5)

actors, counts = zip(*top_actors)
plt.bar(actors, counts)
plt.xlabel('Actor')
plt.ylabel('Movie Count')
plt.title('Top Actors by Movie Count')
plt.show()

# Movie Count by Genre
genre_counts = Counter(genres)

genres, counts = zip(*genre_counts.items())
plt.bar(genres, counts)
plt.xlabel('Genre')
plt.ylabel('Movie Count')
plt.title('Movie Count by Genre')
plt.xticks(rotation=45)
plt.show()

# Duration Distribution
plt.hist(movie_durations, bins=20)
plt.xlabel('Duration (minutes)')
plt.ylabel('Frequency')
plt.title('Duration Distribution')
plt.show()

# Distribution of Description Lengths
description_lengths = [len(movie['description']) for movie in movie_database]
plt.hist(description_lengths, bins=20)
plt.xlabel('Description Length')
plt.ylabel('Frequency')
plt.title('Distribution of Description Lengths')
plt.show()

# Average Movie Duration by Genre
genre_durations = {}
for movie in movie_database:
    if movie['genre']:
        for genre in movie['genre']:
            if genre not in genre_durations:
                genre_durations[genre] = []
            genre_durations[genre].append(duration_to_minutes(movie['duration']))

average_durations = {genre: sum(durations) / len(durations) for genre, durations in genre_durations.items()}

plt.bar(average_durations.keys(), average_durations.values())
plt.xlabel('Genre')
plt.ylabel('Average Duration (minutes)')
plt.title('Average Movie Duration by Genre')
plt.xticks(rotation=45)
plt.show()

# Movie Count by Actor
actor_counts = Counter([actor for movie in movie_database for actor in movie['actor']])

actors, counts = zip(*actor_counts.most_common(5))
plt.bar(actors, counts)
plt.xlabel('Actor')
plt.ylabel('Movie Count')
plt.title('Movie Count by Actor')
plt.xticks(rotation=45)
plt.show()
