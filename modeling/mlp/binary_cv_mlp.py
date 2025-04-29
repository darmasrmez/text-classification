import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

def load_data(file_path):
    """Load the dataset from a CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(df, ngram_range=(1, 1)):
    """Preprocess the data using CountVectorizer(binary=True) and LabelEncoder."""
    vectorizer = CountVectorizer(binary=True, ngram_range=ngram_range)
    X_counts = vectorizer.fit_transform(df['Feature']).toarray()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['Section'])

    return X_counts, y, label_encoder

def split_data(X, y, test_size=0.2, random_state=0):
    """Split the data into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=True)

def train_and_evaluate(X_train, X_test, y_train, y_test, label_encoder):
    """Train the MLP model and evaluate its performance."""
    pipe = Pipeline([
        ('classifier', MLPClassifier(hidden_layer_sizes=(200, 100), max_iter=300, random_state=0))
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    print("Classification Report for MLP with CountVectorizer(binary=True)")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

def binary_cv_mlp():
    """Main function to execute the workflow."""
    file_path = '/home/darmas/IPN/7/NLP/text_classification/modeling/arxiv_preprocessed.csv' 

    df = load_data(file_path)
    X, y, label_encoder = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    train_and_evaluate(X_train, X_test, y_train, y_test, label_encoder)
