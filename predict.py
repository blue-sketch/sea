from preprocess import preprocess_text  # Import the function from preprocess.py




def predict_sentiment(model, tfidf, review):
    review_cleaned = preprocess_text(review)
    review_tfidf = tfidf.transform([review_cleaned])
    prediction = model.predict(review_tfidf)
    return "Positive" if prediction == 1 else "Negative"
