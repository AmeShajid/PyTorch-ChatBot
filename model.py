#First make a virtual environement for this project
#python3 -m venv venv
#next download pytorch pip3 install torch torchvision torchaudio
#then pip install nltk 
#then make an intents page that includes all of our greets and responses
#then make the nltk_utils page and activate pytorch
#after finishing the nlk then make a train.py
#next make this model.py and we are making the model
#after train.py is done make a chat.py

# Import the PyTorch library, which helps us build and train neural networks
import torch
# Import the neural network module from PyTorch
import torch.nn as nn

# Define a new class called NeuralNet that will create our neural network model
class NeuralNet(nn.Module):
    
    # This function sets up our neural network with layers and activation functions
    def __init__(self, input_size, hidden_size, num_classes):
        # Call the __init__ function of the parent class to initialize it
        super(NeuralNet, self).__init__()
        
        # Create the first layer that connects input data to the hidden layer
        self.l1 = nn.Linear(input_size, hidden_size)
        # Create the second layer that connects the first hidden layer to another hidden layer
        self.l2 = nn.Linear(hidden_size, hidden_size)
        # Create the third layer that connects the second hidden layer to the output
        self.l3 = nn.Linear(hidden_size, num_classes)
        
        # Define a function to add activation (make the model learn more complex patterns)
        self.relu = nn.ReLU()
    
    # This function defines how data moves through the layers of the network
    def forward(self, x):
        # Pass the input data through the first layer
        out = self.l1(x)
        # Apply the activation function to the output of the first layer
        out = self.relu(out)
        # Pass the result through the second layer
        out = self.l2(out)
        # Apply the activation function again
        out = self.relu(out)
        # Pass the result through the third layer to get the final output
        out = self.l3(out)
        # No activation function at the end because we usually apply activation or softmax later during training
        return out






