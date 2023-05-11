#Evaluation scripts: These files contain code that evaluates the performance of the machine learning models using various metrics, 
# such as accuracy, precision, recall, or F1 score. They often take the predicted results and the true labels as input and output the evaluation metrics.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# load the dataset
df = pd.read_csv("C:\\Users\\Martina\\PycharmProjects\\KDD_Social_Biking\\data\\hour.csv")

# separate the features and target variable
X = df.drop(['instant', 'dteday', 'cnt'], axis=1)
y = df['cnt']

# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# train a logistic regression model
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)

# make predictions on the testing set
y_pred = lr.predict(X_test_scaled)

# calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
f1 = f1_score(y_test, y_pred, average='weighted')


# print the results
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1)
