import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk

# Download required resourses from NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    # Lowersing
    text = text.lower()
    
    # Removing Punctuction
    text = text.translate(str.maketrans("","", string.punctuation))
    
    # Removing numbers
    text = re.sub(r'\d+','', text)
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Removing stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Stemming (Optional)
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    
    # Joining tokens back to text
    processed_text =  ' '.join(tokens)
    
    return processed_text

# Example usage
raw_text = "The quick brown fox jumps over the lazy dog! It was an amazing sight to behold in 2023."
processed_text = preprocess_text(raw_text)
print("Original Text: ", raw_text)
print("Processed Text: ", processed_text)
