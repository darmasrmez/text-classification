from rich.console import Console
from rich.markdown import Markdown

from svm.binary_cv_svm import binary_svm
from svm.cv_svm import cv_svm
from svm.tf_idf_svm import tf_idf_svm

from naive_bayes.cv_nb import cv_svm
from naive_bayes.binary_cv_nb import binary_nb
from naive_bayes.tf_idf_nb import tf_idf_nb

if __name__ == "__main__":
    console = Console()
    md_content = """
# Running Models
This script runs the SVM, Naive Bayes and MLP models using the next representations for text data:

- CountVectorizer
- CountVectorizer(binary=True)
- tf-idf

The models are trained and evaluated on the preprocessed ArXiv dataset.

## Results
The results of the models are printed in the console. The classification report includes precision, recall, and F1-score for each class.
"""
    md = Markdown(md_content)
    console.print(md)
    console.print(Markdown("## SVM"))
    print("\nRunning SVM with CountVectorizer")
    cv_svm()
    print("\nRunning SVM with CountVectorizer(binary=True)")
    binary_svm()
    print("\nRunning SVM with tf-idf")
    tf_idf_svm()
    console.print(Markdown("## Naive Bayes"))
    print("\nRunning Naive Bayes with CountVectorizer")
    cv_svm()
    print("Running Naive Bayes with CountVectorizer(binary=True)")
    binary_nb()
    print("\nRunning Naive Bayes with tf-idf")
    tf_idf_nb()
    print("All models have been run successfully.")