# TASK 1A
**Harkeerat Singh Sawhney**

## Introduction to the task:
In Task 1A we were supposed to perform 10-fold cross-validation with ridge regression through each given value of λ and then to perform RMSE over and then to take its average over the 10 tests fold. We are supposed to train the ridge regression while leaving out a different fold each time.

## Approach to the solution:
The approach for this was to follow the sample solution given and to implement the solution where it was not implemented. Overall the solution was implemented by primarily following the sklearn website (which contains documentation for different methods used in the solution)

## Methods used in the solution:
### `fit()`:
The first function, fit(), takes in training input points, input labels, and a lambda parameter. The function returns the optimal parameters for ridge regression. The function creates a Ridge Regression object using the Ridge() function. The lambda parameter is used along with the fit_intercept parameter set to false. This parameter is used to avoid overfitting, thus reducing the variance of the model. This change to the Ridge function was found online, where it is a common choice.

### `calculate_RMSE()`:
The second function, calculate_RMSE(), takes in test data points and computes the Root Mean Square Error (RMSE) of the prediction done by the fit() method. The function uses the mean_squared_error() method from the sklearn library to compute the Mean Square Error. The value of MSE is then square rooted to obtain RMSE.

### `average_LR_RMSE()`:
The third function, average_LR_RMSE(), implements 10-fold cross-validation (CV) as mentioned in the sample solution. The function creates a KFold object that splits the dataset 10 number of pieces. The shuffle parameter is set to true, and the random_state parameter is set to 4. These parameters were chosen to reduce bias and improve the generalization of the model. The function uses nested for loops. For each value of the lambda, the fit() function is used to obtain the optimal parameters of the Ridge Regression. The calculate_RMSE() function is applied with the obtained optimal parameters w on the test data. The average RMSE across the 10 folds is computed and returned. 

## SOURCES:
-	https://www.javatpoint.com/rsme-root-mean-square-error-in-python (Better understanding of RMSE)
-	https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html (Sklearn documentation on KFold)
