# SUML Delivery project series
# By Hryhorii Hrymailo s27157

from sklearn.datasets import load_iris
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os


def main():
    # Import dataset
    iris = load_iris()
    X, y = iris.data, iris.target_names[iris.target]

    # Create train/test datasets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define and train model
    # model = DummyClassifier(strategy="most_frequent")
    # model = LogisticRegression()
    model = RandomForestClassifier()

    model.fit(X_train, y_train)

    # Run model on test dataset
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Dump model.joblib
    os.makedirs("app", exist_ok=True)
    joblib.dump(model, "app/model.joblib")
    print("Model saved to /app/model.joblib")

if __name__ == "__main__":
    main()
