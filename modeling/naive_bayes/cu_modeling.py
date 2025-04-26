from cuml.model_selection import train_test_split
import cudf
import cupy as cp
from cuml.preprocessing import LabelEncoder
from cuml.feature_extraction.text import CountVectorizer, TfidfVectorizer
from cuml.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from cuml.pipeline import Pipeline
import numba
from numba import cuda

df = cudf.read_csv('./arxiv_preprocessed.csv')


vectorizer = TfidfVectorizer()
X_counts = vectorizer.fit_transform(df['Feature'])
X_cupy = cp.asarray(X_counts.toarray())

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(df['Section'])


X_train, X_test, y_train, y_test = train_test_split(X_cupy, y, test_size=0.2, random_state=0, shuffle=True)

# Built a pipeline with MultinomialNB
pipe = Pipeline([
    ('classifier', MultinomialNB())
])

# Fit the pipeline on the training data
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
y_test_cpu = y_test.to_numpy()
y_pred_cpu = y_pred.get()

print(classification_report(y_test_cpu, y_pred_cpu, target_names=label_encoder.classes_.to_pandas()))