import argparse
import mlflow
import mlflow.sklearn
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def train_model(model_name):
    # Load dataset
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_name == "elasticnet":
        model = ElasticNet(alpha=0.5, l1_ratio=0.5)
    elif model_name == "knn":
        model = KNeighborsRegressor(n_neighbors=5)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Train model
    model.fit(X_train, y_train)

    # Predict and evaluate
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    # Log experiment
    mlflow.set_experiment("Model Evaluation")
    with mlflow.start_run():
        mlflow.log_param("model", model_name)
        mlflow.log_metric("mse", mse)
        mlflow.sklearn.log_model(model, "model")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model to train")
    args = parser.parse_args()
    train_model(args.model)

if __name__ == "__main__":
    main()
