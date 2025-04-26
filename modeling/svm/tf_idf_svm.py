import cudf
import cupy as cp
import numba

from numba import cuda
from cuml.model_selection import train_test_split
from cuml.preprocessing import LabelEncoder
from cuml.feature_extraction.text import TfidfVectorizer
from cuml.svm import SVC
from sklearn.metrics import classification_report
from cuml.pipeline import Pipeline


def load_data(file_path):
    """Load the dataset from a CSV file."""
    return cudf.read_csv(file_path)


def preprocess_data(df, ngram_range=(1, 1)):
    """Preprocess the data using CountVectorizer and LabelEncoder."""
    vectorizer = TfidfVectorizer(ngram_range=ngram_range)
    X_counts = vectorizer.fit_transform(df['Feature'])
    X_cupy = cp.asarray(X_counts.toarray())

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['Section'])

    return X_cupy, y, label_encoder


def split_data(X, y, test_size=0.2, random_state=0):
    """Split the data into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=True)


def train_and_evaluate(X_train, X_test, y_train, y_test, label_encoder):
    """Train the MultinomialNB model and evaluate its performance."""
    pipe = Pipeline([
        ('classifier', SVC())
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    y_test_cpu = y_test.to_numpy()
    y_pred_cpu = y_pred.get()

    print("Classification Report for SVM with tf-idf")
    print(classification_report(y_test_cpu, y_pred_cpu, target_names=label_encoder.classes_.to_pandas()))


def tf_idf_svm():
    """Main function to execute the workflow."""
    file_path = '/home/darmas/IPN/7/NLP/text_classification/modeling/arxiv_preprocessed.csv'

    # Load data
    df = load_data(file_path)

    # Preprocess data
    X, y, label_encoder = preprocess_data(df)

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train and evaluate
    train_and_evaluate(X_train, X_test, y_train, y_test, label_encoder)