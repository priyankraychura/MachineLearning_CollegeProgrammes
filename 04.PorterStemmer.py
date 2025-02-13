from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk

# Create a Porter Stemmer instance
porter_stemmer = PorterStemmer()

# Example words for stemming
words = ["running", "jumps", "happily", "running", "happily"]

# Apply steamming to each word
stemmed_words = [porter_stemmer.stem(word) for word in words]

# Print the result
print("Original words: ", words)
print("Stemmed words: ", stemmed_words)

nltk.download('punkt_tab')
# Initialize the Porter Stemmer
stemmer = PorterStemmer()

# Example sentence
sentence = "The quick brown foxes were jumping over the lazy dogs."

# Tokenize the sentence
words = word_tokenize(sentence)

# Stemming the words
stemmed_words = [stemmer.stem(word) for word in words]

# Print the stemmed words
print("Original words: ", words)
print('Stemmed words: ', stemmed_words)