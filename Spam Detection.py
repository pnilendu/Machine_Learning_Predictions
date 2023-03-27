import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Download the dataset from: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
# Save the file as 'spam.csv' in the current working directory
df = pd.read_csv(r'D:\Datasets\spam.csv', encoding='ISO-8859-1', on_bad_lines='skip')
if 'Unnamed: 2' in df.columns:
    df = df.drop(['Unnamed: 2'], axis=1)
if 'Unnamed: 3' in df.columns:
    df = df.drop(['Unnamed: 3'], axis=1)
if 'Unnamed: 4' in df.columns:
    df = df.drop(['Unnamed: 4'], axis=1)
df = df.rename(columns={'v1': 'label', 'v2': 'text'})  # Rename columns for better readability
# Check if 'label' column exists in DataFrame
if 'label' not in df.columns:
    raise KeyError("Column 'label' not found in DataFrame")
df['label'] = np.where(df['label'] == 'spam', 1, 0)  # Convert labels to binary values (0 for 'ham', 1 for 'spam')

# Clean the text data
nltk.download('stopwords')
ps = PorterStemmer()
corpus = []
for i in range(len(df)):
    text = df['text'][i].lower()  # Convert to lowercase
    text = re.sub('[^a-zA-Z0-9]', ' ', text)  # Remove special characters
    text = text.split()  # Tokenize
    text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]  # Stemming and removing stopwords
    text = ' '.join(text)
    corpus.append(text)
df['text'] = corpus

# Create Bag of Words and TF-IDF models for feature extraction
cv = CountVectorizer()
X = cv.fit_transform(df['text']).toarray()
tv = TfidfVectorizer()
X_tfidf = tv.fit_transform(df['text']).toarray()
y = df['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_tfidf_train, X_tfidf_test, y_tfidf_train, y_tfidf_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=0)

# Fitting the Naive Bayes classifier to the Bag of Words model
nb = MultinomialNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
print('Accuracy of Naive Bayes with Bag of Words:', accuracy_score(y_test, y_pred))
print('Confusion Matrix of Naive Bayes with Bag of Words:', confusion_matrix(y_test, y_pred))
print('Classification Report of Naive Bayes with Bag of Words:', classification_report(y_test, y_pred))

# Fitting the Naive Bayes classifier to the TF-IDF model
nb_tfidf = MultinomialNB()
nb_tfidf.fit(X_tfidf_train, y_tfidf_train)
y_tfidf_pred = nb_tfidf.predict(X_tfidf_test)
print('Accuracy of Naive Bayes with TF-IDF:', accuracy_score(y_tfidf_test, y_tfidf_pred))
print('Confusion Matrix of Naive Bayes with TF-IDF:', confusion_matrix(y_tfidf_test, y_tfidf_pred))
print('Classification Report of Naive BayBayes with TF-IDF:', classification_report(y_tfidf_test, y_tfidf_pred))

#Fitting the Random Forest classifier to the Bag of Words model
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
y_pred_rfc = rfc.predict(X_test)
print('Accuracy of Random Forest with Bag of Words:', accuracy_score(y_test, y_pred_rfc))
print('Confusion Matrix of Random Forest with Bag of Words:', confusion_matrix(y_test, y_pred_rfc))
print('Classification Report of Random Forest with Bag of Words:', classification_report(y_test, y_pred_rfc))

#Fitting the Random Forest classifier to the TF-IDF model
rfc_tfidf = RandomForestClassifier()
rfc_tfidf.fit(X_tfidf_train, y_tfidf_train)
y_tfidf_pred_rfc = rfc_tfidf.predict(X_tfidf_test)
print('Accuracy of Random Forest with TF-IDF:', accuracy_score(y_tfidf_test, y_tfidf_pred_rfc))
print('Confusion Matrix of Random Forest with TF-IDF:', confusion_matrix(y_tfidf_test, y_tfidf_pred_rfc))
print('Classification Report of Random Forest with TF-IDF:', classification_report(y_tfidf_test, y_tfidf_pred_rfc))

#Fitting the XGBoost classifier to the Bag of Words model
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
print('Accuracy of XGBoost with Bag of Words:', accuracy_score(y_test, y_pred_xgb))
print('Confusion Matrix of XGBoost with Bag of Words:', confusion_matrix(y_test, y_pred_xgb))
print('Classification Report of XGBoost with Bag of Words:', classification_report(y_test, y_pred_xgb))

#Fitting the XGBoost classifier to the TF-IDF model
xgb_tfidf = XGBClassifier()
xgb_tfidf.fit(X_tfidf_train, y_tfidf_train)
y_tfidf_pred_xgb = xgb_tfidf.predict(X_tfidf_test)
print('Accuracy of XGBoost with TF-IDF:', accuracy_score(y_tfidf_test, y_tfidf_pred_xgb))
print('Confusion Matrix of XGBoost with TF-IDF:', confusion_matrix(y_tfidf_test, y_tfidf_pred_xgb))
print('Classification Report of XGBoost with TF-IDF:', classification_report(y_tfidf_test, y_tfidf_pred_xgb))