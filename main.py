import pandas as pd
from preprocess import preprocess_text
from model import train_model
from evaluate import evaluate_model
from predict import predict_sentiment

# Load and preprocess data
df = pd.read_csv("C:\\Users\\Neel\\Downloads\\IMDB Dataset.csv.zip")
df = df.sample(1000)
df['cleaned_review'] = df['review'].apply(preprocess_text)

# Train the model
model, tfidf, X_test_tfidf, y_test = train_model(df)

# Evaluate the model
evaluate_model(model, X_test_tfidf, y_test)

# Predict sentiment for new data
new_review = "This movie was fantastic!"
print("Sentiment:", predict_sentiment(model, tfidf, new_review))
