

IMDb Movie Review Sentiment Analysis





Project Overview:
This project is a machine learning-based sentiment analysis system that predicts whether a movie review is positive or negative. 
The system is built using the IMDb dataset, which consists of thousands of movie reviews labeled with their sentiment. The model is trained using natural language processing (NLP) techniques and machine learning algorithms.

Features
Data Preprocessing: The text data is preprocessed by tokenization, stopword removal, lemmatization, and transforming into numerical features using TF-IDF.

Model Training: A Logistic Regression model is trained to classify reviews as positive or negative.

Model Evaluation: The model is evaluated using accuracy, confusion matrix, and classification reports to ensure its performance.

Prediction: The model can predict the sentiment of new reviews based on the patterns it learned from the dataset.


HOW IT WORKS:

Data Loading: The IMDb dataset is loaded and a sample of the data is used for training and testing purposes.

Text Preprocessing: Reviews are cleaned and preprocessed, including lowercasing, tokenization, stopword removal, and lemmatization.

Feature Extraction: Reviews are transformed into numerical features using the TF-IDF vectorizer.

Model Training: The Logistic Regression model is trained on the preprocessed data.

Model Evaluation: The model's performance is evaluated using accuracy, confusion matrix, and classification report.

Prediction: The system can predict the sentiment of new reviews by preprocessing them and using the trained model.

RESULTS:

Accuracy: 85%

Confusion Matrix:

True Negatives: 478

False Positives: 122

False Negatives: 95

True Positives: 605

Precision, Recall, F1-Score: Detailed classification report for both positive and negative reviews.
