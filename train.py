
#import json because we are working with a json file
import json
#import this is a natural reading languge library basically process and puts it english
import nltk
#download this
#nltk.download('punkt_tab')
#this is for mathematical operations on arrays
import numpy as np
#from our other file import this
from nltk_utils import tokenize, stem, bag_of_words

# import the core PyTorch library for building and training models
import torch

# import the neural network module, which contains classes and functions for building neural networks
import torch.nn as nn

# import tools for creating datasets and data loaders, which help manage and organize data for training models
from torch.utils.data import Dataset, DataLoader

#this is from the model file
from model import NeuralNet


#here we are opening the intent.json file(mine only worked with the path not the name idk why and open as read)
with open('/Users/ameshajid/Documents/VisualStudioCode/Small Projects/Pytorch-Chatbot/intent.json', 'r') as f:
    #then the intents is going to be loaded in f which is our file
    intents = json.load(f)

#this is going to store all the words 
all_words = []
#this is going to store all the intents tag like (greeting, goodbye and wtvr)
tags = []
#this will hold the pattern and texts
xy = []

#we use intents because in the json we have our intents key that has everything in it
for intent in intents['intents']:
    #to get our tag we get it from the intents json and the tag part
    tag = intent['tag']
    #then we add that to our tag variable
    tags.append(tag)
    #for the patterns in patterns
    for pattern in intent["patterns"]:
        #we will array all of them break them up
        w = tokenize(pattern)
        #we will extend the arry and not append
        all_words.extend(w)
        #we are appending a tuple so its appened not extend, and we are addding both tag and word
        xy.append((w, tag))

#these are the characters we want to ignore
ignore_words = ["?", "!", ".", ","]

# Create a new list by processing each word in the original all_words list
# Only include words that are not in the ignore_words list
# Apply stemming to each of these words
all_words = [stem(w) for w in all_words if w not in ignore_words]
"""
longer way:
# Create a list to hold stemmed words
processed_words = []
# Iterate through each word in the all_words list
for word in all_words:
    # Check if the current word is not in the list of words to ignore
    if word not in ignore_words:
        # Apply the stemming function to the word to reduce it to its root form
        stemmed_word = stem(word)
        # Add the stemmed word to the processed_words list
        processed_words.append(stemmed_word)
# Now we are sorting these words
print(sorted(processed_words))
"""
# Remove duplicate words by converting the list to a set
# Then sort the unique words in alphabetical order
# Finally, convert the sorted set back to a list
all_words = sorted(set(all_words))

#doing the same thing to tags
tags = sorted(set(tags))

#our bag of words
# Create empty lists to store the features (X_train) and labels (Y_train)
X_train = []
Y_train = []

# Go through each pattern_sentence and tag in the xy list
for (pattern_sentence, tag) in xy:
    # Convert the pattern_sentence into a list of numbers using all_words
    # This helps the computer understand the sentence the list is based on the words in allwords
    bag = bag_of_words(pattern_sentence, all_words)
    # Add this list of numbers to X_train (features)
    X_train.append(bag)

    # Find the position of the tag in the tags list
    # This gives us a number that represents the tag
    label = tags.index(tag)
    # Add this number to Y_train (labels)
    Y_train.append(label)

# Turn the list of feature data into a format that’s easier for the computer to work with
# This makes it easier to perform mathematical operations on the data
X_train = np.array(X_train)

# Turn the list of labels into the same format
# This helps in efficiently handling and processing the labels during training
Y_train = np.array(Y_train)

# Hyper-parameters
# These are settings that we choose before training the model

# Number of times we will loop through the entire dataset
num_epochs = 1000

# Number of samples we process at once during training
batch_size = 8

# How fast the model learns (smaller numbers mean slower learning)
learning_rate = 0.001

# The number of features in each piece of training data
input_size = len(X_train[0])

# Number of neurons in each hidden layer
hidden_size = 8

# Number of categories our model can predict
output_size = len(tags)

# Print the input size and output size to check our settings
print(input_size, output_size)


class ChatDataset(Dataset):
    # This is a class that helps PyTorch handle our data

    def __init__(self):
        # We store the number of samples, and the data (inputs and outputs)
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = Y_train

    # This method lets us get a sample from our data using an index
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # This method tells us how many samples we have
    def __len__(self):
        return self.n_samples

# Create a dataset object from our ChatDataset class
dataset = ChatDataset()

# Create a DataLoader to handle our data in batches
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,  # Number of samples per batch
                          shuffle=True,           # Shuffle the data each time
                          num_workers=0)          # Number of workers for loading data (0 means use the main thread)

# Decide whether to use a GPU (faster) or CPU (slower) for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create our neural network model and move it to the chosen device (GPU or CPU)
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Define the loss function (how we measure errors) and optimizer (how we update the model)
criterion = nn.CrossEntropyLoss()                      # Loss function to measure how wrong the model's predictions are
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Optimizer to adjust the model based on errors

# Start training the model
for epoch in range(num_epochs):
    # Go through each batch of data
    for (words, labels) in train_loader:
        # Move data to the chosen device (GPU or CPU)
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass: Get the model's predictions
        outputs = model(words)
        
        # Calculate how wrong the predictions are
        loss = criterion(outputs, labels)
        
        # Backward pass: Calculate the gradients (how to change the model to reduce error)
        optimizer.zero_grad()   # Clear previous gradients
        loss.backward()        # Calculate new gradients
        optimizer.step()       # Update the model with the new gradients
        
    # Print the loss every 100 epochs to see how well the model is learning
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Print the final loss after all epochs
print(f'final loss: {loss.item():.4f}')



# Save the model's state and settings to a file
data = {
    "model_state": model.state_dict(),  # Save the model's learned weights
    "input_size": input_size,           # Save the number of input features
    "hidden_size": hidden_size,         # Save the number of neurons in hidden layers
    "output_size": output_size,         # Save the number of output categories
    "all_words": all_words,             # Save the list of all words
    "tags": tags                        # Save the list of tags (categories)
}

# Save the data to a file named 'data.pth'
FILE = "data.pth"
torch.save(data, FILE)

# Print a message to let us know the training is done and the file is saved
print(f'training complete. file saved to {FILE}')


