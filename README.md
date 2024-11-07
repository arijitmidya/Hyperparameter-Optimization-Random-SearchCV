# Hyperparameter-Optimization-Random-SearchCV

## Steps of Implementation : 

1. Define the hyperparameters for the ML model   
We start by specifying the range of values that each hyperparameter can take. In this step, we need to identify the hyperparameters of the ML algorithm we want to use and their function during the learning process. The easy way to do so is to read the documentation of the machine learning algorithm. For example, the logistic regression algorithm has about 15 parameters that can be combined. You can refer to the documentation provided by the scikit-learn library.

Here are some examples of hyperparameters from logistic regression:

penalty: This specifies the norm of the penalty.

C: This represents the inverse of regularization strength.

dual: This determines if the dual or primal formulation is implemented.

tol: This represents the tolerance for stopping criteria.

class_weight: This represents the weights associated with classes in the form {class_label: weight}.

solver: This represents the algorithm that optimizes the problem.

2. Set the number of iterations for the random search
In the second step, we have to define the total number of iterations to control the number of random hyperparameter combinations that will be evaluated. The default value is 10. The larger the number the better, because the random search method will be able to randomly try different combinations before selecting the best one.
However, it might take a lot of time to optimize because it might require more computational power if the model is complex and the dataset is large.

3. Create random combinations of hyperparameter values
For each iteration, the random search method randomly selects a value from the defined range of values for each hyperparameter. The total number of combinations of hyperparameter values depends on the number of iterations defined.

For example, if the number of iterations is 10, then the random search method will randomly create 10 different combinations of hyperparameter values.

4. Train the ML model using the randomly selected hyperparameters
In this step, the random search method will use the randomly selected hyperparameters to train the ML model on the training data. This process will be repeated for the rest of the combinations of the hyperparameter values.For example, if we set the number of iterations to five, then the ML model is trained on five different combinations of hyperparameters.

5. Evaluate and record the performance of the ML model
The performance of the model will be evaluated on the validation data using a specified evaluation metric based on the ML problem, either classification or regression. When working on the ML project for classification problems, we can use the following evaluation metrics:

Accuracy: This is the number of correct predictions from an ML model divided by the total number of predictions made.

F1 score: This is the harmonic mean of precision and recall.

If we’re working on an ML project for regression problems, we can use the following evaluation metrics:

Mean absolute error (MAE): This is the absolute value of the difference between the predicted value and the actual value.

Root mean squared error (RMSE): This is the square root of the value calculated from the mean squared error (MSE).

Finally, the random search method records the ML model's performance for each iteration based on the evaluation metric specified.

6. Select the best-performing combination of hyperparameters
After going through all the iterations, the random search method selects the combination of hyperparameters that produce the best model performance based on the evaluation metric used in Step 5.

![image](https://github.com/user-attachments/assets/0f853703-c034-438c-913c-d791c3235caf)

In the above image, we can see that the fourth combination of hyperparameters produces an ML model with a best performance of 0.97 compared to other combinations of hyperparameters.

7. Retrain the model using the selected hyperparameters
The final step is to train the final ML model using the selected hyperparameters on the complete training data and validation data. This will be the final ML model with the best performance based on the combination of hyperparameters. The ML model will be used to make predictions on new or unseen data.

These are the important steps that a random search method must follow to find the best combination of hyperparameters to produce an ML model with the best performance. The random search only tries a fixed number of combinations and chooses the best among them.

# As an example Binary Classification problem statement is solved using Logistic regression and Random Forest Algorithm
