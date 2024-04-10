from sklearn.base import BaseEstimator, TransformerMixin
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re
import numpy as np


class TextCleanerWithStemming(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stemmer = PorterStemmer()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([self._get_clean_text_with_stemming(text) for text in X])

    def _get_clean_text_with_stemming(self, text):
        cleaned_text = re.sub(r'[^a-zA-Z ]', ' ', text)
        tokens = word_tokenize(cleaned_text)
        stemmed_tokens = [self.stemmer.stem(word) for word in tokens if len(word) > 1]
        return ' '.join(stemmed_tokens)


class TextLengthTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([len(x) for x in X]).reshape(-1, 1)


class DigitCountTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([sum(c.isdigit() for c in x.replace(' ', '')) for x in X]).reshape(-1, 1)


class UppercaseCountTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([sum(c.isupper() for c in re.findall(r'\w+', x)) for x in X]).reshape(-1, 1)


class URLPresenceTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([1 if re.search(re.compile(r'https?://\S+|www\.\S+'), x) else 0 for x in X]).reshape(-1, 1)
