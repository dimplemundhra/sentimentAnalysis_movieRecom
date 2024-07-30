import pandas as pd

# import nltk
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer

# from sklearn.feature_extraction.text import TfidfVectorizer

# from sklearn.linear_model import LogisticRegression

# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

path = r'C:\Users\DIMPLE MUNDHRA\OneDrive\文档\DataScienceProject_1\IMDB_datasets.csv'
imdb_reviews=pd.read_csv(path)

imdb_reviews.head()

# def preprocess_text(text):
#     # Tokenize the text
#     tokens = word_tokenize(text)
    
#     # Remove stopwords
#     stop_words = set(stopwords.words('english'))
#     tokens = [t for t in tokens if t not in stop_words]
    
#     # Remove punctuation
#     tokens = [t for t in tokens if t.isalpha()]
    
#     # Lemmatize the tokens
#     lemmatizer = WordNetLemmatizer()
#     tokens = [lemmatizer.lemmatize(t) for t in tokens]
    
#     # Join the tokens back into a string
#     text = ' '.join(tokens)
    
#     return text

# def extract_features(texts):
#     # Create a TF-IDF vectorizer
#     vectorizer = TfidfVectorizer()
    
#     # Fit the vectorizer to the text data and transform it into a matrix
#     X = vectorizer.fit_transform(texts)
    
#     return X

# def build_model(X, y):
#     # Create a logistic regression model
#     model = LogisticRegression()
    
#     # Train the model on the data
#     model.fit(X, y)
    
#     return model

# def evaluate_model(model, X, y):
#     # Predict the sentiments using the model
#     y_pred = model.predict(X)
    
#     # Calculate the accuracy, precision, recall, and F1-score
#     accuracy = accuracy_score(y, y_pred)
#     precision = precision_score(y, y_pred)
#     recall = recall_score(y, y_pred)
#     f1 = f1_score(y, y_pred)
    
#     return accuracy, precision, recall, f1