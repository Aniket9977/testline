import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler


# Function to train the model
def create_and_train_model(data_path, model_path="model.pkl"):

    # Load the dataset
    df = pd.read_csv(data_path)

    # Ensure the required columns are present
    required_columns = ["Temperature", "Run_Time", "Downtime_Flag"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Dataset must contain the columns: {required_columns}")

    # Separate features and target variable
    X = df[["Temperature", "Run_Time"]]
    y = df["Downtime_Flag"]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features (optional but recommended)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Select and train the model
    model = DecisionTreeClassifier()

    model.fit(X_train, y_train)

    # Save the trained model and scaler
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return {"accuracy": accuracy, "f1_score": f1}


# Example usage
if __name__ == "__main__":
    # Define file paths
    DATA_PATH = "synthetic_machine_data.csv"
    MODEL_PATH = "model.pkl"

    # Train the model and print metrics
    metrics = create_and_train_model(DATA_PATH, model_path=MODEL_PATH)
    print("Model training complete!")
    print("Performance Metrics:", metrics)
