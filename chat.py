


# Import the libraries we need
import random  # For picking random responses
import json    # For working with JSON files
import torch   # For working with neural networks

# Import our neural network model and helper functions
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

# Set up to use GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Open and read our intents JSON file which has example questions and answers
with open('/Users/ameshajid/Documents/VisualStudioCode/Small Projects/Pytorch-Chatbot/intent.json', 'r') as json_data:
    intents = json.load(json_data)
# Load the saved model data from a file
FILE = "/Users/ameshajid/Documents/VisualStudioCode/Small Projects/Pytorch-Chatbot/data.pth"
#This change tells PyTorch to only load the model's weights and not any additional data that might execute code, making it safer.
data = torch.load(FILE, weights_only=True)

# Extract information needed to use the model
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

# Create the neural network model and load the saved data into it
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()  # Set the model to evaluation mode

# Name of the chatbot
bot_name = "Sam"
print("Let's chat! (type 'quit' to exit)")

while True:
    # Get input from the user
    sentence = input("You: ")
    
    # If the user types 'quit', end the chat
    if sentence == "quit":
        break

    # Tokenize the sentence (split it into words)
    sentence = tokenize(sentence)
    
    # Convert the tokenized sentence into a format the model understands
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])  # Make sure it's the right shape
    X = torch.from_numpy(X).to(device)  # Convert it to a tensor and move it to the device

    # Get the model's prediction
    output = model(X)
    _, predicted = torch.max(output, dim=1)  # Get the index of the highest score

    # Get the tag associated with the highest score
    tag = tags[predicted.item()]

    # Get the probabilities for each tag
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    
    # If the probability is high enough, respond with a random response for the tag
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I do not understand...")



