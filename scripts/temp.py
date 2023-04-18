# Segregating data based on ‘workingday’ column
work_day = df[df['workingday'] == 1]
non_work_day = df[df['workingday'] == 0]
# Model for registered
x = work_day.drop(['casual', 'registered', 'count'], axis=1)
y = work_day.registered

# Dividing the data into train and test
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=2)

import statsmodels.formula.api as smf
model1=smf.OLS(y_train,x_train)
result = model1.fit()
result.summary()
y_pred = result.predict(x_test)

from sklearn.metrics import mean_squared_log_error
msle=mean_squared_log_error(y_pred,y_test)
rmsle=np.sqrt(msle)
print('RMLSE for the data:',rmsle)

# Finding best parameters for decision tree
dt = DecisionTreeRegressor(random_state=0)
dt_params = {‘max_depth’:np.arange(1,50,2),’min_samples_leaf’:np.arange(2,15)}

from sklearn.model_selection import GridSearchCV
gs_dt = GridSearchCV(dt,dt_params,cv=3)
gs_dt.fit(x_train,y_train)
a = gs_dt.best_params_

# Training with best parameters
from sklearn.tree import DecisionTreeRegressor
dtr=DecisionTreeRegressor(max_depth=a[‘max_depth’],min_samples_leaf= a[‘min_samples_leaf’])
model = dtr.fit(x_train,y_train)
y_pred = model.predict(x_test)

from sklearn.metrics import mean_squared_log_error
msle=mean_squared_log_error(y_pred,y_test)
rmsle=np.sqrt(msle)
print('RMLSE for the data:',rmsle) # For decision tree

# Finding best parameters for RandomForestRegressor
rf = RandomForestRegressor(random_state=0)
rf_params = {‘n_estimators’:np.arange(25,150,25),’max_depth’:np.arange(1,11,2),’min_samples_leaf’:np.arange(2,15,3)}

from sklearn.model_selection import GridSearchCV
gs_rf = GridSearchCV(rf,rf_params,cv=3)
gs_rf.fit(x_train,y_train)
b = gs_rf.best_params_

# Fitting the model with best params
RF = RandomForestRegressor(n_estimators=b['n_estimators'],max_depth=b['max_depth'],min_samples_leaf=b['min_samples_leaf'],random_state=0)
model = RF.fit(x_train,y_train)
y_pred = model.predict(x_test)

from sklearn.metrics import mean_squared_log_error
msle=mean_squared_log_error(y_pred,y_test)
rmsle=np.sqrt(msle)
print('RMLSE for the data:',rmsle) # For random forest

# Finding best parameters for AdaBoostRegressor
ar = AdaBoostRegressor(base_estimator=RF,random_state=0)
ar_params = {‘n_estimators’:np.arange(25,200,25)}
from sklearn.model_selection import GridSearchCV
gs_ar = GridSearchCV(ar,ar_params,cv=3)
gs_ar.fit(x_train,y_train)
c = gs_ar.best_params_

# Fitting the model with best params
ab_rf = AdaBoostRegressor(base_estimator=RF,n_estimators=c['n_estimators'],random_state=0)
model = ab_rf.fit(x_train,y_train)
y_pred = model.predict(x_test)

from sklearn.metrics import mean_squared_log_error
msle=mean_squared_log_error(y_pred,y_test)
rmsle=np.sqrt(msle)
print('RMLSE for the data:',rmsle) # For Ada-Boost



#------------------------------------------------------------------------------------------------------