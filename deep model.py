import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pymongo import MongoClient
import torch
from torch import nn
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, Lambda, Subtract
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K


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


def create_full_data_collection():
    full_data = db['data_for_NN']
    new_collection = db['train_set']
    full_data = [mov for mov in full_data.find()]

    movies = db['movies']
    for index, d in enumerate(full_data):
        try:
            data = {"vector_1": movies.find_one({"_id": d["movie1"]})["bert-base-nli-mean-tokens-embedding"],
                    "vector_2": movies.find_one({"_id": d["movie2"]})["bert-base-nli-mean-tokens-embedding"],
                    "similar": d["similar"]}
            print(index)
            new_collection.insert_one(data)
        except:
            continue


def data_generator(db_col, batch_size):
    while True:
        batch_data = db_col.aggregate([{'$sample': {'size': batch_size}}])
        vectors_1 = []
        vectors_2 = []
        labels = []
        for d in batch_data:
            vectors_1.append(d['vector_1'])
            vectors_2.append(d['vector_2'])
            labels.append(d['similar'])
        yield [np.array(vectors_1), np.array(vectors_2)], np.array(labels)


def model_1():
    data = db['train_set']
    batch_size = 32
    num_samples = data.count_documents({})
    train_indices, test_indices = train_test_split(range(num_samples), test_size=0.2, random_state=42)
    train_generator = data_generator(data, batch_size)
    test_generator = data_generator(data, batch_size)
    train_steps_per_epoch = len(train_indices) // batch_size
    test_steps_per_epoch = len(test_indices) // batch_size
    model = Sequential([
        Dense(512, activation='relu', input_shape=(768 * 2,)),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_generator, epochs=10, steps_per_epoch=train_steps_per_epoch, validation_data=test_generator,
              validation_steps=test_steps_per_epoch, verbose=2)

    test_loss, test_acc = model.evaluate(test_generator, steps=test_steps_per_epoch)
    print('Test accuracy:', test_acc)


def model_2():
    collection = db['train_set']

    # Load data from MongoDB collection
    data = []
    for entry in collection.find().limit(32):
        data.append(entry)

    # Extract vectors and labels
    vectors_1 = []
    vectors_2 = []
    labels = []
    for entry in data:
        vectors_1.append(entry['vector_1'])
        vectors_2.append(entry['vector_2'])
        labels.append(entry['similar'])

    # Convert to numpy arrays
    X = np.array([vectors_1, vectors_2]).transpose((1, 0, 2))
    y = np.array(labels)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define input layers
    input_1 = Input(shape=(768,))
    input_2 = Input(shape=(768,))

    # Define shared layers
    shared_layer_1 = Dense(512, activation='relu')
    shared_layer_2 = Dense(256, activation='relu')
    shared_layer_3 = Dense(128, activation='relu')

    # Define feature extraction subnetworks
    encoded_1 = shared_layer_3(shared_layer_2(shared_layer_1(input_1)))
    encoded_2 = shared_layer_3(shared_layer_2(shared_layer_1(input_2)))

    # Define distance function using L1 distance
    distance = Lambda(lambda x: K.abs(x[0] - x[1]))([encoded_1, encoded_2])

    # Define output layer
    output = Dense(1, activation='sigmoid')(distance)

    # Define model
    model = Model(inputs=[input_1, input_2], outputs=output)

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])

    # Train model on training data
    model.fit([X_train[:, 0], X_train[:, 1]], y_train,
              batch_size=8,
              epochs=10)

    # Evaluate model on test data
    loss, accuracy = model.evaluate([X_test[:, 0], X_test[:, 1]], y_test)
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)
