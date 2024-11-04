import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import nltk

# Ensure the necessary NLTK resources are downloaded
nltk.download('stopwords', force=True)
nltk.download('punkt', force=True)

# Load dataset
data_path = r"C:\Users\Aravind\hate_speech_detection\HateSpeechDataset.csv"  # Using raw string

data = pd.read_csv(data_path)

# Display first 5 rows and columns
print("First 5 rows of the dataset:")
print(data.head())
print("\nColumns in the dataset:", data.columns)

# Preprocess text function
def preprocess_text(text):
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    import string

    # Lowercase the text
    text = text.lower()
    # Tokenize the text
    words = word_tokenize(text)
    # Remove stopwords and punctuation
    words = [word for word in words if word.isalnum()]
    return ' '.join(words)

# Apply preprocessing
data['processed_text'] = data['Content'].apply(preprocess_text)

# Prepare features and labels
X = data['processed_text']  # Text data
y = data['Label']   # Assuming 'class' column contains the labels

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Predict on test set
y_pred = model.predict(X_test_vectorized)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Function to classify custom input
def classify_text(input_text):
    # Preprocess input
    processed_input = preprocess_text(input_text)
    # Vectorize input
    input_vectorized = vectorizer.transform([processed_input])
    # Predict class
    prediction = model.predict(input_vectorized)[0]
    return prediction

# User input for classification
while True:
    user_input = input("Enter a text to classify (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    result = classify_text(user_input)
    if result == 0:
        print("The text is classified as: Hate Speech")
    elif result == 1:
        print("The text is classified as: Offensive Language")
    else:
        print("The text is classified as: Neutral")
