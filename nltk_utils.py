#how this words:
#sentence = we get our sentence: What would you do?
#Tokenize it =["what", "would", "you", "do", "?"] 
#stem it and lowercase everything =["what", "would", "you", "do"]  
#then it goes into the bag of words = [0, 0, 1, 0](example one)

#import this is a natural reading languge library basically process and puts it english
import nltk
#download this then comment it out once its downloaded
#nltk.download('punkt')

import numpy as np
#this is so we can grab the stemmer
from nltk.stem.porter import PorterStemmer
#this is creating the stemmer machine
stemmer = PorterStemmer()


#So in this function we basically split up the string into units
#splits everythign into an array including punctuation numbers characters
#"What would you do?" = ["what", "would", "you", "do", "?"]
def tokenize(sentence):
    #this tokenizes every singel world
    return nltk.word_tokenize(sentence)


#This generates the root form of the words
#basically chops off the ends of words and gets the roots
#[organize, organizes, organizing] = [organ, organ, organ]
def stem(word):
    #this basically stems the word and makes everything lowercase
    return stemmer.stem(word.lower())


#it converts the tokenizied sentence into a bag of words(binary list)
#the binary list shows which words from our all words array in the intents.json it matches up with which gives a 90% similarity rate
def bag_of_words(tokenized_sentence, all_words):
    """
    This function creates a 'bag of words' which is a list of 0s and 1s. 
    Each position in the list corresponds to a word from 'all_words'. 
    If a word from 'all_words' is in the sentence, that position becomes 1, otherwise, it stays 0.
    
    Example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", hello", "I", "you", "bye", "thank", "cool"] 
    Resulting bag = [0, 1, 0, 1, 0, 0, 0]
    """
    
    # Create an empty list to store stemmed words
    stemmed_words = []
    
    # Loop through each word in the tokenized sentence
    for word in tokenized_sentence:
        # Stem the word and add it to the stemmed_words list
        stemmed_word = stem(word)
        stemmed_words.append(stemmed_word)
    
    # Create a list of zeros, with the same length as all_words
    bag = np.zeros(len(all_words), dtype=np.float32)
    
    # Loop through each word in all_words
    for idx, word in enumerate(all_words):
        # Check if the word is in the stemmed_words list
        if word in stemmed_words:
            # If found, set the corresponding position in the bag to 1
            bag[idx] = 1.0
            
    # Return the final bag of words
    return bag
