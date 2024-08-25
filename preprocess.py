import nltk

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')



from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def preprocess_text(text):
    text = text.lower()
    words = word_tokenize(text)  # Use word_tokenize which requires 'punkt'
    words = [word for word in words if word.isalpha() and word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)


