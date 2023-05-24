#Prediction scripts: These files contain code that makes predictions or classifications on new data using the trained models. 
# They often take new data files as input and output the predicted results.
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

def data_cleaner(data_frame):
    data_frame["season"].replace({1:4, 2:1, 3:2, 4:3}, inplace=True)
    columns_to_drop = ['instant', 'dteday', 'mnth', 'holiday', 'weekday', 'temp', 'casual', 'registered']
    data_frame = data_frame.drop(columns=columns_to_drop)
    return data_frame

def outlier_cleaner_atemp(data_frame):
    upper_bound = data_frame['atemp'].quantile(0.95)
    data_frame_no_outliers = data_frame[(data_frame['atemp'] <= upper_bound)]
    return(data_frame_no_outliers)

def outlier_cleaner_wind(data_frame):
    upper_bound = data_frame['windspeed'].quantile(0.95)
    data_frame_no_outliers = data_frame[(data_frame['windspeed'] <= upper_bound)]
    return (data_frame_no_outliers)

def outlier_cleaner_hum(data_frame):
    lower_bound = data_frame['hum'].quantile(0.05)
    data_frame_no_outliers = data_frame[(lower_bound <= data_frame['hum'])]
    return (data_frame_no_outliers)

def outlier_cleaner(dataframe):
    o1 = outlier_cleaner_atemp(dataframe)
    o2 = outlier_cleaner_wind(o1)
    o3 = outlier_cleaner_hum(o2)
    return (o3)


def correlation(data_frame):
    correlation_matrix = data_frame.corr()
    correlation_with_target = correlation_matrix['cnt'].sort_values(ascending=False)
    print(correlation_with_target)
    return correlation_with_target

def split_train_test(data_frame):
    # Split into features (X) and target variable (y)
    X = data_frame.drop('cnt', axis=1)  # Drop the 'cnt' column from the features
    y = data_frame['cnt']  # Select the 'cnt' column as the target variable

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def LR(X_train, X_test, y_train):
    # Create an instance of the LinearRegression modellr
    modellr = LinearRegression()

    # Fit the modellr to the training data
    modellr.fit(X_train, y_train)

    # Make predictions on the testing data
    y_predlr = modellr.predict(X_test)
    return y_predlr

def DTR(X_train, X_test, y_train):
    modeldt = DecisionTreeRegressor()
    modeldt.fit(X_train, y_train)
    y_pred = modeldt.predict(X_test)
    return y_pred

def RFR(X_train, X_test, y_train):
    modelrf = RandomForestRegressor()
    modelrf.fit(X_train, y_train)
    y_pred = modelrf.predict(X_test)
    return y_pred

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

