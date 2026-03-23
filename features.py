from sklearn.feature_extraction.text import TfidfVectorizer

def compute_tfidf(text_data, max_features=5000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(text_data)
    return X, vectorizer