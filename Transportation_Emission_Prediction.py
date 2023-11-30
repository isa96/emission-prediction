#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('cd', 'C:\\Users\\ASUS\\Documents\\Dataset_Hackathon\\archive')


# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import pickle
import tensorflowjs as tfjs


# In[52]:


data = pd.read_csv('CO2 Emissions_Canada.csv')


# In[53]:


data.head()


# In[54]:


# data.info()


# In[55]:


# data.describe()


# In[56]:


data.isnull().sum()


# In[57]:


data = data.drop(['Make', 'Model', 'Vehicle Class', 'Transmission', 'Fuel Consumption Comb (L/100 km)', 'Fuel Consumption Comb (mpg)'], axis=1)
data.head()


# # EDA

# In[58]:


fuel_type = data['Fuel Type']


# In[59]:


fuel_type.value_counts()


# In[65]:


# one hot encode
encode = {}

new_value = {'Z':'premium gasoline',
            'X' : 'regular gasoline',
            'D' : 'diesel',
            'E' : 'ethanol (E85)',
            'N' : 'natural gas'}
# for index in range (len(np.unique(fuel_type))):
#     encode.update({fuel_type.value_counts().index[index] : i})
#     i += 1
    
onehot = {'Fuel Type': new_value}
data.replace(onehot, inplace=True)


# In[66]:


fuel_type.value_counts()


# In[67]:


data.head()


# In[68]:


y_axis = fuel_type.value_counts()
x_axis = ['Regular Gasoline', 'Premium Gasoline', 'Diesel', 'Ethanol (E85)', 'Natural Gas']

fig = plt.figure(figsize = (10, 5))
sns.barplot(x =x_axis , y=y_axis, data=data).set(title='Fuel Type')
plt.show()


# In[69]:


numerical_data = ['Engine Size(L)','Cylinders',  'Fuel Consumption City (L/100 km)', 'Fuel Consumption Hwy (L/100 km)']


# In[70]:


fig = plt.figure(figsize=(30,20))

for index in range(len(numerical_data)):
    ax = fig.add_subplot(2,3,index+1)
    ax.set_title('correlation between CO2 Emissions(g/km) and {}'.format(numerical_data[index]))
    ax.scatter( data[numerical_data[index]], data['CO2 Emissions(g/km)'])
plt.show()


# In[71]:


np.unique(fuel_type)


# In[15]:


# # one hot encode
# encode = {}
# i = 0
# for index in range (len(np.unique(fuel_type))):
#     encode.update({fuel_type.value_counts().index[index] : i})
#     i += 1
    
# onehot = {'Fuel Type': encode}
# data.replace(onehot, inplace=True)


# In[72]:


onehot = pd.get_dummies(data, columns=['Fuel Type'])


# In[73]:


onehot.head()


# In[74]:


data_clean = onehot


# In[75]:


x = data_clean.drop(['CO2 Emissions(g/km)'], axis=1)
y = data_clean['CO2 Emissions(g/km)']


# In[76]:


x


# In[77]:


# Show correlation between each features using heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = data_clean.corr().round(1)

#To print score in the box, use anot=True parameter
sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, )
plt.title("Correlation Matrix for Numeric Feature ", size=20)


# In[78]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)


# # Modelling

# In[155]:


mae = []
accuracy = []


# ## Linear Regression

# In[80]:


lr_parameters = {'fit_intercept':[True, False],
                'normalize': [True, False],
                'n_jobs': [10, 30, 50, 70, 90, 100]}

grid = RandomizedSearchCV(LinearRegression(),
                   lr_parameters)

grid.fit(x_train, y_train)
print('best estimator: {}'.format(grid.best_estimator_))
print('best parameters: {}'.format(grid.best_params_))
print('best score: {}'.format(grid.best_score_))


# In[81]:


lr_model = LinearRegression(fit_intercept=False, n_jobs=30).fit(x_train, y_train)


# In[82]:


lr_predict = lr_model.predict(x_test)


# In[156]:


lr_r2_accuracy = r2_score(y_test, lr_predict)
lr_mae = mean_absolute_error(y_test, lr_predict)

accuracy.append(lr_r2_accuracy)
mae.append(lr_mae)
print('Linear Regression Accuracy: {}'.format(lr_r2_accuracy))
print('Linear Regression MAE: {}'.format(lr_mae))
print('Linear Regression Coeffiecient: {} '.format(lr_model.coef_))
print('Linear Regression Intercept: {} '.format(lr_model.intercept_))


# In[84]:


# prediction = lr_model.predict([[2.0,4, 9.9,6.7,0,0,0,0,1]])


# ## Ridge Regression

# In[85]:


ridge_parameters = {'fit_intercept':[True, False],
                 'alpha':[1.0, 2.0, 3.0, 4.0, 5.0],
                'normalize': [True, False],
                'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}

grid = RandomizedSearchCV(Ridge(),
                   ridge_parameters)

grid.fit(x_train, y_train)
print('best estimator: {}'.format(grid.best_estimator_))
print('best parameters: {}'.format(grid.best_params_))
print('best score: {}'.format(grid.best_score_))


# In[86]:


ridge_model = Ridge(alpha=2.0, solver='svd').fit(x_train, y_train)


# In[87]:


ridge_predict = ridge_model.predict(x_test)


# In[157]:


ridge_r2_score = r2_score(y_test, ridge_predict)
ridge_mae = mean_absolute_error(y_test, ridge_predict)
accuracy.append(ridge_r2_score)
mae.append(ridge_mae)

print('Ridge Accuracy: {}'.format(ridge_r2_score))
print('Ridge MAE: {}'.format(ridge_mae))
print('Ridge Coeffiecient: {} '.format(ridge_model.coef_))
print('Ridge Intercept: {} '.format(ridge_model.intercept_))


# # Random Forest Regressor

# In[89]:


forest_parameters = {'n_estimators':[100, 200, 300, 400, 500],
                     'max_depth':[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                     'min_samples_split':[2,4,6,8,10],
                     'min_samples_leaf':[1,3,5,7,9],
                     'max_features': ['sqrt', 'log2', None]
                }

grid = RandomizedSearchCV(RandomForestRegressor(),
                   forest_parameters)

grid.fit(x_train, y_train)
print('best estimator: {}'.format(grid.best_estimator_))
print('best parameters: {}'.format(grid.best_params_))
print('best score: {}'.format(grid.best_score_))


# In[90]:


random_forest = grid.best_estimator_.fit(x_train, y_train)


# In[91]:


random_forest_predict = random_forest.predict(x_test)


# In[158]:


random_forest_r2_score = r2_score(y_test, random_forest_predict)
random_forest_mae = mean_absolute_error(y_test, random_forest_predict)
accuracy.append(random_forest_r2_score)
mae.append(random_forest_mae)

print('Random Forest Regressor Accuracy: {}'.format(random_forest_r2_score))
print('Random Forest Regressor MAE: {}'.format(random_forest_mae))


# # Neural Network

# In[93]:


from keras.callbacks import EarlyStopping
callback = EarlyStopping(monitor = 'loss',
                        patience = 2,
                        verbose = 1)


# In[146]:


nn_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(60, input_shape = [9], activation ='relu'),
    tf.keras.layers.Dense(60, activation='relu'),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dense(3, activation='relu'),
#     tf.keras.layers.Dropout(0,3),
    tf.keras.layers.Dense(1, activation = 'linear')
])

nn_model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3),
                loss= 'mean_absolute_error',
                metrics = ['mae'])

nn_model.fit(x_train, y_train, epochs = 500)


# In[160]:


nn_model_predict = nn_model.predict(x_test)


# In[162]:


nn_r2_score = r2_score(y_test,nn_model_predict)
nn_mae = mean_absolute_error(y_test,nn_model_predict)

accuracy.append(nn_r2_score)
mae.append(nn_mae)

print('Neural Network Accuracy: {}'.format(nn_r2_score))
print('Neural Network MAE: {}'.format(nn_mae))


# # Evaluate Model

# In[163]:


evaluate_dict = {'accuracy': accuracy,
                 'mae': mae}

evaluate_model = pd.DataFrame(evaluate_dict)


# In[164]:


rows_name = ['Logistic_Regression', 'Ridge_Regression', 'Random_Forest', 'Neural_Network']


# In[165]:


evaluate_model.index = rows_name


# In[166]:


evaluate_model


# In[167]:


x_axis= np.arange(len(rows_name))

plt.figure(figsize=(20,18))
plt.bar(x_axis - 0.2, accuracy, 0.4, label='accuracy')
plt.bar(x_axis + 0.2, mae, 0.4, label = 'mae')

plt.xticks(x_axis, rows_name)
plt.xlabel("Evaluate Model")
plt.legend()
plt.show()


# # Save Model

# In[150]:


pickles = ['linear_regression_model.pkl', 'ridge_regression_model.pkl','random_forest_model.pkl' ]
ml_models = [lr_model, ridge_model, random_forest]


# In[151]:


for index in range (len(pickles)):
    with open(pickles[index], 'wb') as file:
        pickle.dump(ml_models[index], file)


# In[154]:


#test load random forest
with open('random_forest_model.pkl', 'rb') as file:
    load_model = pickle.load(file)
    
load_predict = load_model.predict(x_test)
print(r2_score(y_test, load_predict))


# In[ ]:


nn_model.save('nn_regression.h5')


# In[23]:


# convert neural network model into tfjs
tfjs.converters.save_keras_model(nn_model, 'C:/Users/ASUS/Documents/Dataset_Hackathon/Transportation_Emission_Prediction/models/tfjs_model')


# # Prediction Model

# In[102]:


fuel = ['Regular Gasoline', 'Premium Gasoline','Ethanol (E85)', 'Diesel', 'Natural Gas']


# In[140]:


reg = nn_model.predict([[2.0,4, 9.9,6.7,0,0,0,0,1]])
premium = nn_model.predict([[2.0,4, 9.9,6.7,0,0,0,1,0]])
ethanol = nn_model.predict([[2.0,4, 9.9,6.7,0,0,1,0,0]])
diesel = nn_model.predict([[2.0,4, 9.9,6.7,0,1,0,0,0]])
natural = nn_model.predict([[2.0,4, 9.9,6.7,1,0,0,0,0]])


# In[141]:


print(reg)
print(premium)
print(ethanol)
print(diesel)
print(natural)


# In[ ]:




