# Overview

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; In this era, emission is the most important thing that we must concern. With high emission there is a lot impact that we can feel, there are air pollution, climate change, etc. if emissions from human activities increase, they build up in the atmosphere and warm the climate, leading to many other changes around the world—in the atmosphere, on land, and in the oceans.as emissions from human activities increase, they build up in the atmosphere and warm the climate, leading to many other changes around the world—in the atmosphere, on land, and in the oceans [Climate Change Indicator](https://www.epa.gov/climate-indicators/greenhouse-gases). and this is can be a big problem for us as a human.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; One of factors that can produced a lot of emissions is transportation, therefore this notebook wants to predict emission from transportation to help reduce high emission vehicle. 

# Exploratory Data Analysis

1. Use **data.info** to see information of each columns and we know that there are 73585 rows and 12 columns
2. use **data.isnull().sum()** to check null or missing values in dataset
3. Because we only need several columns like Engine Size, Cylinders, Fuel Type, Fuel Consumption City, Fuel Consumption Highway (Hwy) and CO2 Emissions(g/km), then we remove the rest using **data = data.drop([colums], axis=1)**

![image](https://user-images.githubusercontent.com/91602612/202978110-7928e4a9-5fb2-43d0-a805-2610cda22905.png)

4. To make user can see the fuel type meaning we change the alphabet representation using actual fuel type

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; from: ![image](https://user-images.githubusercontent.com/91602612/202978264-468f6ee2-8a37-4237-b7d2-f378f020d41c.png)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; to: ![image](https://user-images.githubusercontent.com/91602612/202978302-22fa1bed-eadd-4eb9-acdd-878010619ab2.png)

5. Visualize total of each fuel type

![image](https://user-images.githubusercontent.com/91602612/202978452-4d137921-37bb-4fd9-9c3b-a7bbe1cf7540.png)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Then we know that regular gasoline is the highest fuel type that most vehicles use and natural gas is the least fuel type that vehicle use

6. Then we plot correlation for each numerical data using scatter

![image](https://user-images.githubusercontent.com/91602612/202978738-d4211173-fa92-4514-ac69-51c746f78ede.png)

# Data Preprocessing

1. Because there is fuel type column that contain non numerical value, therefore we need to encode that into numerical value using **pd.get_dummies()**

![image](https://user-images.githubusercontent.com/91602612/202979945-de27392c-3910-4701-a6fb-c8c16fd12b92.png)

2. Define x values and y value for x values contain all independent variables and y values contain label or dependent variable
3. split x and y into x_train, x_test, y_train, y_test using **train_test_split** and in this case I use split size 80% for train_size and 20% for test_size

# Modeling

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; For modeling I use 4 model, there are:
1. Linear Regression with estimator **LinearRegression(fit_intercept=False, n_jobs=30)**
2. Ridge Regression with estimator **Ridge(alpha=2.0, solver='svd')**
3. Random Forest Regression with estimator **RandomForestRegressor(max_depth=50, max_features=None, min_samples_split=8)**
4. Neural Network with layers like this:

![image](https://user-images.githubusercontent.com/91602612/202980391-626ed1e3-9643-492a-8a0f-5a5bf36d9509.png)

# Result

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; For the result I got accuracy and MAE for each model like this:

![image](https://user-images.githubusercontent.com/91602612/202980543-f041eeae-0810-4d57-9d44-c49684dee04e.png)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Then we can see that best MAE and accuracy goes to Random Forest Regression. Not only that, I alos saved my models into pickle and js for tensorflow or deep learning model



