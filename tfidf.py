import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
nltk.download('stopwords')

# Import data using pandas and create two dataframes with only stars and text.
all_columns = ['review_id', 'user_id', 'business_id', 'stars', 'date', 'text', 'useful', 'funny', 'cool']
good_columns = ['stars', 'text']
# Can change good stars to include all 1-5 or select certain stars only
good_stars = ['1', '3', '5']
sample_size = 10000

data = pd.read_json('data/review-0-0.json')
data = data[data['stars'].isin(good_stars)]
# Selecting only the first 10000 rows for computational time and memory error
# MemoryError: Unable to allocate array with shape (100000, 113946) and data type float64
x = data.head(sample_size)['text']
y = data.head(sample_size)['stars']

# Cleaning the data to remove puncutation, stop words, then returns the cleaned text
def clean_text(text):
    not_punc = [char for char in text if char not in string.punctuation]
    not_punc = ''.join(not_punc)
    return [word.lower() for word in not_punc.split() if word.lower() not in stopwords.words('english')]

    
# Converts the data into a vector
tfidf_vectorizer = TfidfVectorizer(use_idf=True)
X = tfidf_vectorizer.fit_transform(x).toarray()


# Split the dataset into train and test data
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=101)

# Tests the baseline accuracy using a support vector machine classifier
sv_classifier = SVC(random_state=101, gamma='scale')
sv_classifier.fit(x_train, y_train)
predict_svm = sv_classifier.predict(x_test)

# Prints a classification report and the accuracy score
print("Classification Report:",classification_report(y_test,predsvm))
print("Score:",round(accuracy_score(y_test,predsvm)*100,2))