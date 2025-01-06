import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from src.preprocessing import load_preprocessed_data
import time as t

def train_and_evaluate(model_class, dataframe, X, y, split):
    """
    Train, test, and evaluate a scikit-learn model using RMSE.

    Parameters:
        model_class: The scikit-learn model class to use (e.g., sklearn.ensemble.RandomForestRegressor).
        dataframe: The pandas DataFrame containing the dataset.
        X: List of column names to use as predictors.
        y: The name of the target column.
        split: Float, percentage of the dataset to use for training (e.g., 0.8 for 80%).

    Returns:
        model: The trained model.
        rmse: Root Mean Squared Error on the test set.
    """
    start = t.time()
    print(f"Training on {model_class}...")

    # Split the dataset into training and testing sets
    X_data = dataframe[X]
    y_data = dataframe[y]
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=(1 - split), random_state=42)
    print("Training data shape: ", X_train.shape)
    print("Test data shape: ", X_test.shape)

    # Initialize and train the model
    print('fitting the model..')
    model = model_class()
    model.fit(X_train, y_train)

    # Make predictions
    print('predicting..')
    start_prediction = t.time()
    y_pred = model.predict(X_test)
    prediction_time = t.time() - start_prediction

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"Root Mean Squared Error (RMSE) for {model_class}: {rmse}")

    end = t.time()

    print(f'Finished in {end-start} seconds. Prediction time for 20% of the data: {prediction_time}')

    return model, rmse

def optimize_randomforest(dataframe, X, y, split):
    """
    Fine-tune RandomForestRegressor hyperparameters using RandomizedSearchCV and report the best RMSE achieved.

    Parameters:
        dataframe: The pandas DataFrame containing the dataset.
        X: List of column names to use as predictors.
        y: The name of the target column.
        split: Float, percentage of the dataset to use for training (e.g., 0.8 for 80%).

    Returns:
        best_model: The optimized RandomForestRegressor model.
        best_rmse: The best RMSE achieved during optimization.
    """
    print("Optimizing RandomForestRegressor...")

    X_data = dataframe[X]
    y_data = dataframe[y]
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=(1 - split), random_state=42)

    param_distributions = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [2, 4],
        'max_features': ['log2']
    }

    model = RandomForestRegressor(random_state=42)

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=30,
        scoring='neg_mean_squared_error',
        cv=3,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )

    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_
    print(f"Best Parameters: {random_search.best_params_}")

    y_pred = best_model.predict(X_test)
    best_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"Best Root Mean Squared Error (RMSE): {best_rmse}")

    return best_model, best_rmse

if __name__ == "__main__":
    data = load_preprocessed_data()
    target = "prec"
    #model, rmse = train_and_evaluate(RandomForestRegressor, data, [col for col in data.columns if col != target], target,0.7)
    #model, rmse = train_and_evaluate(GradientBoostingRegressor, data, [col for col in data.columns if col != target], target,0.7)
    #best_model, best_rmse = optimize_randomforest(data, [col for col in data.columns if col != target], target, 0.7)

    import matplotlib.pyplot as plt

    # Data
    models = ['RandomForest', 'GradientBoost', 'NeuralNet', 'LinearModel']
    heights = [2.5, 3.6, 2.6, 2.7]

    # Create bar chart
    plt.bar(models, heights)
    for i, height in enumerate(heights):
        plt.text(i, height + 0.1, str(height), ha='center')
    # Labeling the graph
    plt.title('Model Comparison on Inference Time')
    plt.xlabel('Models')
    plt.ylabel('RMSE')

    # Display the graph
    plt.show()
