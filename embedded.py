from pymongo import MongoClient
import torch.optim as optim
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel, GPT2Tokenizer, AlbertTokenizer, AlbertModel, TransfoXLTokenizer, \
    TransfoXLModel


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


def embedded_bert(batch_size=1):
    movies_collection = db['movies']

    print("Querying the database to get the descriptions")
    data = movies_collection.find({}, {"description": 1})

    descriptions = []
    for doc in data:
        descriptions.append(doc["description"])

    print(f'Found {len(descriptions)} movies')

    # Load the BERT model and tokenizer
    model_name = "sentence-transformers/bert-base-nli-mean-tokens"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    embeddings = []
    for i in range(0, len(descriptions), batch_size):
        # get batch of sentences
        batch = descriptions[i:i + batch_size]

        print(f"Computing embeddings for batch {i // batch_size + 1}")

        # tokenize batch of sentences and move to CUDA
        tokens = tokenizer.batch_encode_plus(batch,
                                             max_length=512,
                                             truncation=True,
                                             padding='max_length',
                                             return_tensors='pt').to(device)

        # disable gradient calculations
        with torch.no_grad():
            # get embeddings for batch and move to CUDA
            outputs = model(**tokens)
            batch_embeddings = outputs.last_hidden_state.to(device)

        # apply attention mask and compute mean-pooled embeddings
        attention_mask = tokens['attention_mask']
        mask = attention_mask.unsqueeze(-1).expand(batch_embeddings.size()).float()
        masked_embeddings = batch_embeddings * mask
        summed = torch.sum(masked_embeddings, 1)
        summed_mask = torch.clamp(mask.sum(1), min=1e-9)
        mean_pooled = summed / summed_mask
        embeddings.append(mean_pooled)

    # concatenate embeddings for all batches and move to CPU
    embeddings = torch.cat(embeddings, dim=0).detach().cpu().numpy()

    print("Updating the documents in the movies_collection with the embeddings")
    # Update the documents in the movies_collection with the embeddings
    all_movies = movies_collection.find({}, {"_id": 1})
    for i, doc in enumerate(all_movies):
        id_ = doc["_id"]
        embedding = embeddings[i]
        movies_collection.update_one({"_id": id_}, {"$set": {f"{model_name} - embedding": embedding.tolist()}})

    return embeddings


# Examples: "xlnet-base-cased", "gpt2", "albert-base-v2", 'transfo-xl-wt103', "bert-large-uncased"
def more_embedded(model_name, batch_size=64):
    movies_collection = db['movies']

    print("Querying the database to get the descriptions")
    data = movies_collection.find({}, {"description": 1})

    descriptions = []
    for doc in data:
        descriptions.append(doc["description"])

    print(f'Found {len(descriptions)} movies')

    # Load the XLNet model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    embeddings = []
    for i in range(0, len(descriptions), batch_size):
        # get batch of sentences
        batch = descriptions[i:i + batch_size]

        print(f"Computing embeddings for batch {i // batch_size + 1}")

        # tokenize batch of sentences and move to device
        tokens = tokenizer(batch,
                           max_length=512,
                           padding='max_length',
                           truncation=True,
                           return_tensors='pt').to(device)

        # disable gradient calculations
        with torch.no_grad():
            # get embeddings for batch and move to device
            outputs = model(**tokens)
            batch_embeddings = outputs.last_hidden_state.to(device)

        # apply attention mask and compute mean-pooled embeddings
        attention_mask = tokens['attention_mask']
        mask = attention_mask.unsqueeze(-1).expand(batch_embeddings.size()).float()
        masked_embeddings = batch_embeddings * mask
        summed = torch.sum(masked_embeddings, 1)
        summed_mask = torch.clamp(mask.sum(1), min=1e-9)
        mean_pooled = summed / summed_mask
        embeddings.append(mean_pooled)

    # concatenate embeddings for all batches and move to CPU
    embeddings = torch.cat(embeddings, dim=0).detach().cpu().numpy()

    print("Updating the documents in the movies_collection with the embeddings")
    # Update the documents in the movies_collection with the embeddings
    for i, doc in enumerate(movies_collection.find({}, {"_id": 1})):
        id_ = doc["_id"]
        embedding = embeddings[i]
        movies_collection.update_one({"_id": id_}, {"$set": {f"{model_name}-embedding": embedding.tolist()}})

    return embeddings


def embedded_lstm():
    # Define hyperparameters
    input_size = 10
    hidden_size = 20
    num_layers = 2
    output_size = 1
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 10

    # Create LSTM model
    lstm_model = LSTMModel(input_size, hidden_size, num_layers, output_size)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learning_rate)

    # Generate some dummy data
    x = torch.randn(100, 20, input_size)
    y = torch.randn(100, output_size)

    # Train the LSTM model
    for epoch in range(num_epochs):
        for i in range(0, len(x), batch_size):
            # Get batch of data
            x_batch = x[i:i + batch_size]
            y_batch = y[i:i + batch_size]

            # Zero out gradients
            optimizer.zero_grad()

            # Forward pass
            y_pred = lstm_model(x_batch)
            loss = criterion(y_pred, y_batch)

            # Backward pass
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        output = self.fc(last_output)
        return output
