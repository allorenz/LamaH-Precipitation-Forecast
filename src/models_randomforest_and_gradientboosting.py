import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

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
    # Split the dataset into training and testing sets
    X_data = dataframe[X]
    y_data = dataframe[y]
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=(1 - split), random_state=42)

    # Initialize and train the model
    model = model_class()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"Root Mean Squared Error (RMSE): {rmse}")

    return model, rmse


if __name__ == "__main__":
    data = getdata()
