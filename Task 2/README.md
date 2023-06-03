# TASK 2
**Harkeerat Singh Sawhney**

## Introduction to the task:
In task 2 we are supposed to apply ML in the real-world problems. The real-world problem in this case is to predict the electricity prices in Switzerland given price information of some other countries and additional features.

## Methods used in the solution:
### `data_loading()`:
First, we load the data from the csv file and put into a pandas dataframe. After that the data is preprocessed by replacing the categorical values in the `season`column with numerical values. This made it easier to work with. After that we are using the K-Nearest Neighbor (KNN) imuter so that we can fill in the missing values. This is done by first fiting the imputer on the training data and then later transforms both the training and the test data using the `transform`method. At last we are splitting the data into the features and the target variable.

### `modeling_and_prediction()`:


This method is taking 3 arguments which are `X_train`, `y_traing`and `X_test`. Initially I tried using linear regression, but that gave me a very low accuracy score. Instead as sugested in the assignment I used the Guassian Naive Bayes Model to fit the training data and then to make the prediciton on the test data. I tried multiple different kernal, but the best one which I got was the RationalQuadratic which my partener figured out.