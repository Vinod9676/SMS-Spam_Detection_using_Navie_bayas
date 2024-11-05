# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
# Replace 'sms_spam.csv' with your dataset file path.
# The dataset should have columns 'label' and 'message'.
data = pd.read_csv('spam.csv', encoding='latin-1')
data = data[['v1', 'v2']]  # Ensure only relevant columns

# Encode labels: 'ham' as 0 and 'spam' as 1
data['v1'] = data['v1'].map({'ham': 0, 'spam': 1})

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['v2'], data['v1'], test_size=0.2, random_state=1)

# Convert text data into numerical data (feature extraction)
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# intializing the navibias
nb_model = MultinomialNB()

#training the model
nb_model.fit(X_train_counts, y_train)

#prediction
y_pred = nb_model.predict(X_test_counts)


print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
