#Data preprocessing scripts: These files contain code that performs data cleaning, feature engineering, data normalization, and other preprocessing tasks. 
#They often take the raw data files as input and output the cleaned and preprocessed data files that can be used for modeling.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load a CSV file
daydf = pd.read_csv('C:/Users/ASUS/PycharmProjects/pythonProject/KDD_Social_Biking/data/day.csv')
hourdf = pd.read_csv('C:/Users/ASUS/PycharmProjects/pythonProject/KDD_Social_Biking/data/hour.csv')

# Print the dataset
#print(daydf)

# Check for missing values
#missing_values_count = daydf.isnull().sum()

# Print the result
#print(missing_values_count)

#There is no missing value for daydf:)

# Check for duplicates
#duplicates = daydf.duplicated()
#print(duplicates)

# Print the duplicated rows
#print(daydf[duplicates])

#There is no duplicated rows

#columns_to_drop = ['instant', 'dteday', 'mnth', 'holiday', 'weekday', 'temp', 'casual', 'registered']
#daydf = daydf.drop(columns=columns_to_drop)
#print(daydf)

def data_cleaner(data_frame):
    columns_to_drop = ['instant', 'dteday', 'mnth', 'holiday', 'weekday', 'temp', 'casual', 'registered']
    data_frame = data_frame.drop(columns=columns_to_drop)
    return data_frame

daydf = data_cleaner(daydf)
hourdf = data_cleaner(hourdf)

lower_bound = hourdf['cnt'].quantile(0.15)
upper_bound = hourdf['cnt'].quantile(0.85)

hourdf_no_outliers = hourdf[(hourdf['cnt'] >= lower_bound) & (hourdf['cnt'] <= upper_bound)]

print(hourdf_no_outliers)

correlation_matrix = daydf.corr()
correlation_with_target = correlation_matrix['cnt'].abs().sort_values(ascending=False)
print(correlation_with_target)


# Split into features (X) and target variable (y)
X = daydf.drop('cnt', axis=1)  # Drop the 'cnt' column from the features
y = daydf['cnt']  # Select the 'cnt' column as the target variable

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Create an instance of the LinearRegression modellr
modellr = LinearRegression()

# Fit the modellr to the training data
modellr.fit(X_train, y_train)

# Make predictions on the testing data
y_predlr = modellr.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_predlr)

# Calculate the root mean squared error
rmse = np.sqrt(mse)

# Calculate the R-squared score
r2 = r2_score(y_test, y_predlr)

# Print the evaluation metrics
print("Mean Squared Error for Linear Regression:", mse)
print("Root Mean Squared Error for Linear Regression:", rmse)
print("R-squared Score for Linear Regression:", r2)
print('\n')



modeldt = DecisionTreeRegressor()
modeldt.fit(X_train, y_train)
y_preddt = modeldt.predict(X_test)
mse = mean_squared_error(y_test, y_preddt)
rmse = np.sqrt(mse)
#mae = mean_absolute_error(y_test, y_preddt)
r2 = r2_score(y_test, y_preddt)
print("Mean Squared Error for Decision Tree:", mse)
print("Root Mean Squared Error for Decision Tree:", rmse)
#print("Mean Absolute Error for Decision Tree:", mae)
print("R-squared for Decision Tree:", r2)
print('\n')

modelrf = RandomForestRegressor()
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

"""model_nn = MLPRegressor(random_state=0)
model_nn.fit(X_train, y_train)
y_pred_nn = model_nn.predict(X_test)
mse_nn = mean_squared_error(y_test, y_pred_nn)
rmse_nn = np.sqrt(mse_nn)
r2_nn = r2_score(y_test, y_pred_nn)
print("Mean Squared Error for Neural Network:", mse_nn)
print("Root Mean Squared Error for Neural Network:", rmse_nn)
print("R-squared for Neural Network:", r2_nn)
print('\n')"""

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






