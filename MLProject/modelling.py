import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import mlflow
import mlflow.sklearn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="heartDisease_preprocessing.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.data_path)
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    mlflow.log_metric("accuracy", report["accuracy"])

    mlflow.sklearn.log_model(model, "model")
    print("Model retraining selesai.")
