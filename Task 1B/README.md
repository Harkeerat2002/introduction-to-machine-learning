# TASK 1B
**Harkeerat Singh Sawhney**

## Introduction to the task:
In Task 1B we are supposed to use 5 different features in a specific order to make our prediction. The way we were supposed to calculate our predictions was through linear structure.

## Approach to the solution:
The approach for this was to follow the sample solution given and to implement the solution where it was not implemented. Overall, the solution was implemented by primarily following the sklearn website (which contains documentation for different methods used in the solution)

## Methods used in the solution:

### `transform_data()`:
The first function, `transform_data(X)`, takes in a matrix X with dimensions (700,5) and transforms each of the 5 input features into 21 new features. Specifically, it creates 5 linear features, 5 quadratic features, 5 exponential features, 5 cosine features, and 1 constant feature. It then returns a new matrix `X_transformed` with dimensions (700,21) containing the transformed data.

### `fit()`:
The second function, `fit(X, y)`, takes in the matrix X and a corresponding label vector y with dimensions (700,). It first calls the `transform_data(X)` function to transform the data, then fits a linear regression model (specifically, a Ridge regression model with alpha= 0.1) on the transformed data. The alpha was found from trial and error, at first it was started from a low value and then kept on increasing. However the lowest and the default value gave the best results. Finally, it returns the optimal parameters w of the linear regression model, which is an array of length 21.

## Sources:
-	https://scikit-learn.org/stable/modules/classes.html#module-sklearn.kernel_ridge (Better understanding of Ridge)
