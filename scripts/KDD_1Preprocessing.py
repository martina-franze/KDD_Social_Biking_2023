#Data preprocessing scripts: These files contain code that performs data cleaning, feature engineering, data normalization, and other preprocessing tasks. 
#They often take the raw data files as input and output the cleaned and preprocessed data files that can be used for modeling.
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense

# Load a CSV file
daydf = pd.read_csv('C:/Users/ASUS/PycharmProjects/pythonProject/KDD_Social_Biking/data/day.csv')
hourdf = pd.read_csv('C:/Users/ASUS/PycharmProjects/pythonProject/KDD_Social_Biking/data/hour.csv')

# Print the dataset
#print(daydf)

# Check for missing values
#missing_values_count = hourdf.isnull().sum()

# Print the result
#print(missing_values_count)

#There is no missing value for daydf:)

# Check for duplicates
#duplicates = hourdf.duplicated()
#print(duplicates)

# Print the duplicated rows
#print(hourdf[duplicates])

#There is no duplicated rows

#columns_to_drop = ['instant', 'dteday', 'mnth', 'holiday', 'weekday', 'temp', 'casual', 'registered']
#daydf = daydf.drop(columns=columns_to_drop)
#print(daydf)

def data_cleaner(data_frame):
    data_frame["season"].replace({1:4, 2:1, 3:2, 4:3}, inplace=True)
    columns_to_drop = ['instant', 'dteday', 'mnth', 'holiday', 'weekday', 'temp', 'casual', 'registered']
    data_frame = data_frame.drop(columns=columns_to_drop)
    return data_frame

daydf = data_cleaner(daydf)
print(daydf)
hourdf = data_cleaner(hourdf)

def outlier_cleaner(data_frame):
#    lower_bound = data_frame['cnt'].quantile(0.25)
#    upper_bound = data_frame['cnt'].quantile(0.75)
#    data_frame_no_outliers = data_frame[(data_frame['cnt'] >= lower_bound) & (data_frame['cnt'] <= upper_bound)]
    upper_bound = data_frame['atemp'].quantile(0.95)
    data_frame_no_outliers = data_frame[(data_frame['atemp'] <= upper_bound)]
    upper_bound = data_frame['hum'].quantile(0.95)
    data_frame_no_outliers = data_frame[(data_frame['hum'] <= upper_bound)]
    lower_bound = data_frame['windspeed'].quantile(0.05)
    data_frame_no_outliers = data_frame[(lower_bound >= data_frame['windspeed'])]
    return data_frame

outlier_cleaner(daydf)
outlier_cleaner(hourdf)

print(daydf)
print(hourdf)

def correlation(data_frame):
    correlation_matrix = data_frame.corr()
    correlation_with_target = correlation_matrix['cnt'].sort_values(ascending=False)
    print(correlation_with_target)
    return correlation_with_target

correlation(daydf)
correlation(hourdf)

def split_train_test(data_frame):
    # Split into features (X) and target variable (y)
    X = data_frame.drop('cnt', axis=1)  # Drop the 'cnt' column from the features
    y = data_frame['cnt']  # Select the 'cnt' column as the target variable

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = split_train_test(hourdf)

def LR(X_train, X_test, y_train):
    # Create an instance of the LinearRegression modellr
    modellr = LinearRegression()

    # Fit the modellr to the training data
    modellr.fit(X_train, y_train)

    # Make predictions on the testing data
    y_predlr = modellr.predict(X_test)
    return y_predlr

y_predlr = LR(X_train, X_test, y_train)

def evaluate(model, y_pred, y_test):
    mse = mean_squared_error(y_test, y_pred)

    # Calculate the root mean squared error
    rmse = np.sqrt(mse)

    # Calculate the R-squared score
    r2 = r2_score(y_test, y_pred)

    # Print the evaluation metrics
    print("Mean Squared Error for ", model, ":", mse)
    print("Root Mean Squared Error for", model, ":", rmse)
    print("R-squared Score for ", model, ":", r2)
    print('\n')

    return mse, rmse, r2

evaluate("Linear Regression", y_predlr, y_test)

def DTR(X_train, X_test, y_train):
    modeldt = DecisionTreeRegressor()
    modeldt.fit(X_train, y_train)
    y_pred = modeldt.predict(X_test)
    return y_pred

y_preddtr = DTR(X_train, X_test, y_train)
evaluate("Decision Tree", y_preddtr, y_test)

def RFR(X_train, X_test, y_train):
    modelrf = RandomForestRegressor()
    modelrf.fit(X_train, y_train)
    y_pred = modelrf.predict(X_test)
    return y_pred

y_predrfr = RFR(X_train, X_test, y_train)
evaluate("Random Forest", y_predrfr, y_test)

def SVRm(X_train, X_test, y_train):
    model_svr = SVR()
    model_svr.fit(X_train, y_train)
    y_pred = model_svr.predict(X_test)
    return y_pred

y_predsvr = SVRm(X_train, X_test, y_train)
evaluate("Support Vector Regression", y_predsvr, y_test)

def gbr(X_train, X_test, y_train):
    model_gb = GradientBoostingRegressor()
    model_gb.fit(X_train, y_train)
    y_pred = model_gb.predict(X_test)
    return y_pred

y_predgbr = gbr(X_train, X_test, y_train)
evaluate("Gradinet Boosting Regression", y_predgbr, y_test)

def knnr(X_train, X_test, y_train):
    model_knn = KNeighborsRegressor(n_neighbors=5)
    model_knn.fit(X_train, y_train)
    y_pred = model_knn.predict(X_test)
    return y_pred

y_predknn = knnr(X_train, X_test, y_train)
evaluate("K-Neighbours Regression", y_predknn, y_test)

#def NeuralNetwork(X_train, X_test, y_train):
#    model_nn = sequential()
#    model_nn.fit(X_train, y_train)
#    y_pred = model_nn.predict(X_test)
#    return y_pred

"""modelrf = RandomForestRegressor()
modelrf.fit(X_train, y_train)
y_predrf = modelrf.predict(X_test)
mse = mean_squared_error(y_test, y_predrf)
rmse = np.sqrt(mse)
#mae = mean_absolute_error(y_test, y_preddt)
r2 = r2_score(y_test, y_predrf)
print("Mean Squared Error for Random Forest:", mse)
print("Root Mean Squared Error for Random Forest:", rmse)
#print("Mean Absolute Error for Decision Tree:", mae)
print("R-squared for Random Forest:", r2)

# Perform cross-validation with 5 folds
scores = cross_val_score(modelrf, X_train, y_train, cv=5, scoring='neg_mean_squared_error')

# Convert negative mean squared error to positive
mse_scores = -scores

# Calculate mean and standard deviation of MSE scores
mean_mse = mse_scores.mean()
std_mse = mse_scores.std()

# Print the mean and standard deviation of MSE scores
print("Mean MSE: ", mean_mse)
print("Standard Deviation of MSE: ", std_mse)
print('\n')

#Detect anomaly data by residue from random forest regression
y_predrf_an = modelrf.predict(X)
residuals = y - y_predrf_an
threshold = 1000
anomalies = daydf[abs(residuals)>threshold]
print(anomalies)
print('\n')

#Detect anomaly data by Clustering

# Step 2: Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3
k = 5 # Number of clusters (you can adjust this based on your dataset)
kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=0)
kmeans.fit(X_scaled)

# Step 4: Assign cluster labels
labels = kmeans.labels_

# Step 5: Calculate distance to centroid for each sample
distances = kmeans.transform(X_scaled)

# Step 6: Define threshold for anomaly detection
threshold = np.percentile(distances, 99)  # Adjust the percentile based on your desired level of anomalies

# Step 7: Detect anomalies
anomalies = daydf[distances.max(axis=1) > threshold]

# Print the anomalies
print(anomalies)





modelgb = GradientBoostingRegressor()
modelgb.fit(X_train, y_train)
y_predgb = modelgb.predict(X_test)
mse = mean_squared_error(y_test, y_predgb)
rmse = np.sqrt(mse)
#mae = mean_absolute_error(y_test, y_preddt)
r2 = r2_score(y_test, y_predgb)
print("Mean Squared Error for Gradient Boosting:", mse)
print("Root Mean Squared Error for Gradient Boosting:", rmse)
#print("Mean Absolute Error for Decision Tree:", mae)
print("R-squared for Gradient Boosting:", r2)

# Perform cross-validation with 5 folds
scores = cross_val_score(modelgb, X_train, y_train, cv=5, scoring='neg_mean_squared_error')

# Convert negative mean squared error to positive
mse_scores = -scores

# Calculate mean and standard deviation of MSE scores
mean_mse = mse_scores.mean()
std_mse = mse_scores.std()

# Print the mean and standard deviation of MSE scores
print("Mean MSE: ", mean_mse)
print("Standard Deviation of MSE: ", std_mse)
print('\n')

#model_nn = MLPRegressor(random_state=0)
#model_nn.fit(X_train, y_train)
#y_pred_nn = model_nn.predict(X_test)
#mse_nn = mean_squared_error(y_test, y_pred_nn)
#rmse_nn = np.sqrt(mse_nn)
#r2_nn = r2_score(y_test, y_pred_nn)
#print("Mean Squared Error for Neural Network:", mse_nn)
#print("Root Mean Squared Error for Neural Network:", rmse_nn)
#print("R-squared for Neural Network:", r2_nn)
#print('\n')

model_svr = SVR()
model_svr.fit(X_train, y_train)
y_pred_svr = model_svr.predict(X_test)
mse_svr = mean_squared_error(y_test, y_pred_svr)
rmse_svr = np.sqrt(mse_svr)
r2_svr = r2_score(y_test, y_pred_svr)
print("Mean Squared Error for Support Vector Regression:", mse_svr)
print("Root Mean Squared Error for Support Vector Regression:", rmse_svr)
print("R-squared for Support Vector Regression:", r2_svr)

# Perform cross-validation with 5 folds
scores = cross_val_score(model_svr, X_train, y_train, cv=5, scoring='neg_mean_squared_error')

# Convert negative mean squared error to positive
mse_scores = -scores

# Calculate mean and standard deviation of MSE scores
mean_mse = mse_scores.mean()
std_mse = mse_scores.std()

# Print the mean and standard deviation of MSE scores
print("Mean MSE: ", mean_mse)
print("Standard Deviation of MSE: ", std_mse)
print('\n')
"""





