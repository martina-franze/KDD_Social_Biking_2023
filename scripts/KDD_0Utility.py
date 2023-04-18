#Utility scripts: These files contain reusable functions or modules that can be used across different parts of the project. 
#They often include helper functions for data cleaning, data transformation, or data visualization.

def model(df):
    # part 4
    # Define global variables
    global nw_count_pred
    global count_pred
    global final_count
    global model_1
    global model_2
    global model_3
    global model_4

    # Split the input dataframe into working day and non-working day dataframes based on the 'workingday' column
    work_day = df[df['workingday'] == 1]
    non_work_day = df[df['workingday'] == 0]

    # Train and fit a random forest regressor model on the working day dataframe for the 'registered' feature
    x = work_day.drop(['casual', 'registered', 'count'], axis=1)
    y = work_day.registered

    # Split the data into training and testing sets
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=2)

    # Find the best parameters for the random forest regressor model using GridSearchCV
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(random_state=0)
    rf_params = {'n_estimators': np.arange(25, 150, 25), 'max_depth': np.arange(1, 11, 2),
                 'min_samples_leaf': np.arange(2, 15, 3)}
    from sklearn.model_selection import GridSearchCV
    gs_rf = GridSearchCV(rf, rf_params, cv=3)
    gs_rf.fit(x_train, y_train)
    a = gs_rf.best_params_

    # Fit the model on the entire working day dataframe using the best parameters
    RF = RandomForestRegressor(n_estimators=a['n_estimators'], max_depth=a['max_depth'],
                               min_samples_leaf=a['min_samples_leaf'], random_state=0)
    model_1 = RF.fit(x, y)



#Utility scripts: These files contain reusable functions or modules that can be used across different parts of the project. 
#They often include helper functions for data cleaning, data transformation, or data visualization.


def predict_bike_rentals(data):
    # original model_test function part 6 
    global predicted_workday_counts
    global predicted_non_workday_counts
    global predicted_counts
    
    # Split the data into workdays and non-workdays based on the 'workingday' column
    workdays = data[data['workingday'] == 1]
    non_workdays = data[data['workingday'] == 0]
    
    # Predict the number of registered users for workdays
    x = workdays
    workday_reg_predictions = trained_model_1.predict(x)
    
    # Predict the number of casual users for workdays
    x = workdays
    workday_casual_predictions = trained_model_2.predict(x)

    # Combine the predicted values for registered and casual users for workdays
    predicted_workday_counts = pd.DataFrame()
    predicted_workday_counts['casual'] = workday_casual_predictions
    predicted_workday_counts['registered'] = workday_reg_predictions
    predicted_workday_counts['total_count'] = workday_casual_predictions + workday_reg_predictions
    predicted_workday_counts.index = x.index

    # Predict the number of registered users for non-workdays
    x = non_workdays
    non_workday_reg_predictions = trained_model_3.predict(x)

    # Predict the number of casual users for non-workdays
    x = non_workdays
    non_workday_casual_predictions = trained_model_4.predict(x)

    # Combine the predicted values for registered and casual users for non-workdays
    predicted_non_workday_counts = pd.DataFrame()
    predicted_non_workday_counts['casual'] = non_workday_casual_predictions
    predicted_non_workday_counts['registered'] = non_workday_reg_predictions
    predicted_non_workday_counts['total_count'] = non_workday_casual_predictions + non_workday_reg_predictions
    predicted_non_workday_counts.index = x.index

    # Concatenate the predicted values for workdays and non-workdays
    predicted_counts = pd.concat([predicted_workday_counts['total_count'], predicted_non_workday_counts['total_count']])
    predicted_counts.sort_index(inplace=True)



