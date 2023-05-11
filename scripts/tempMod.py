# Define a function called `model` which trains and fits a random forest regressor model for the `casual` and `registered` features for both the working and non-working day dataframes.

def model(df):
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

