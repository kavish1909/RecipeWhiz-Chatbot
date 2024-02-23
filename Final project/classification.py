#importing necessary libraries
import pandas as pd
import ast
import time
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# starting time
start = time.time()

df = pd.read_csv("cleaned.csv")           #Path of cleaned dataset

print('Dataset read \n')

df['Features'] = df['Features'].apply(ast.literal_eval)


# Function to filter out non-alphanumeric elements from a set
def filter_set(s):
    return [element for element in s if element.isalnum()]


df['Features'] = df['Features'].apply(filter_set)

print('Column modified to array \n')


# Split dataset into training and testing data
sample_size = 1  # Specify the desired sample size (e.g., 20%)
df_sample = df.sample(frac=sample_size, random_state=42)  # Perform sampling

X = df_sample['Features']
y = df_sample['RecipeCategory']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('Train-test split done. \n')

# Creating the TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fitting and transforming the training data
X_train_df = pd.DataFrame({'Features': X_train.apply(lambda x: ' '.join(x))})
X_train_vectorized = vectorizer.fit_transform(X_train_df['Features'])

# Transforming the testing data
X_test_df = pd.DataFrame({'Features': X_test.apply(lambda x: ' '.join(x))})
X_test_vectorized = vectorizer.transform(X_test_df['Features'])

print('Vectorization done. \n')

# Initializing and fit logistic regression
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train_vectorized, y_train)

# Saving the logistic regression model
joblib.dump(logistic_regression, 'logistic_regression_model.pkl')

# Saving the TF-IDF vectorizer
joblib.dump(vectorizer, 'vectorizer.pkl')

# Predicting on the testing data
y_pred = logistic_regression.predict(X_test_vectorized)

# Displaying actual vs predicted recipe category for 10 rows
df_results = pd.DataFrame({'Actual': y_test[:10], 'Predicted': y_pred[:10]})
print(df_results)

# Evaluating the logistic regression model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1)

end = time.time()
execution_time = end - start
print('Total runtime:', execution_time)
