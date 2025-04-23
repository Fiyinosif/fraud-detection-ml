
# Code Crafters ML Application
import pandas as pd
import joblib
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def load_data():
    base_path = Path(__file__).resolve().parent  # Navigate up from CODE/
    input_train_path = base_path / "INPUT" / "TRAIN" / "fraud_dataset_train.xlsx"
    input_test_path = base_path / "INPUT" / "TEST" / "fraud_dataset_test.xlsx"
    train_df = pd.read_excel(input_train_path)
    test_df = pd.read_excel(input_test_path)
    return train_df, test_df, base_path

def save_results(model_name, model, test_df, test_preds, base_path):
    # Export model
    model_path = base_path / "MODEL" / f"{model_name}_model.joblib"
    joblib.dump(model, model_path)

    # Export predictions
    test_df["Prediction"] = test_preds
    output_path = base_path / "OUTPUT" / f"{model_name}_predictions.csv"
    test_df.to_csv(output_path, index=False)

    # Print evaluation
    print("Test Accuracy:", accuracy_score(test_df['Fraud_Label'], test_preds))
    print("Confusion Matrix:\n", confusion_matrix(test_df['Fraud_Label'], test_preds))
    print("Classification Report:\n", classification_report(test_df['Fraud_Label'], test_preds))

def run_model(model_name, model_class, **kwargs):
    train_df, test_df, base_path = load_data()
    features = ["Amount", "Account_age"]
    target = "Fraud_Label"

    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]

    model = model_class(**kwargs)
    model.fit(X_train, y_train)
    test_preds = model.predict(X_test)
    save_results(model_name, model, test_df, test_preds, base_path)

def load_and_predict(model_name):
    base_path = Path(__file__).resolve().parent.parent
    model_path = base_path / "MODEL" / f"{model_name}_model.joblib"
    input_test_path = base_path / "INPUT" / "TEST" / "fraud_dataset_test.xlsx"
    test_df = pd.read_excel(input_test_path)
    model = joblib.load(model_path)
    features = ["Amount", "Account_age"]
    preds = model.predict(test_df[features])
    print("Reloaded Model Predictions:")
    print(preds)
    print("Accuracy:", accuracy_score(test_df["Fraud_Label"], preds))

def main():
    print("Welcome to Code Crafters Fraud Detection System")
    print("Select an option:")
    print("1 - Train Decision Tree")
    print("2 - Train K-Nearest Neighbors")
    print("3 - Train Support Vector Machine")
    print("4 - Train Artificial Neural Network")
    print("5 - Load Saved Model and Predict")

    choice = input("Enter your choice (1-5): ")

    if choice == '1':
        run_model("decision_tree", DecisionTreeClassifier, random_state=42)
    elif choice == '2':
        run_model("knn", KNeighborsClassifier, n_neighbors=5)
    elif choice == '3':
        run_model("svm", SVC, kernel='rbf', probability=True)
    elif choice == '4':
        run_model("ann", MLPClassifier, hidden_layer_sizes=(10, 5), max_iter=500, random_state=42)
    elif choice == '5':
        model_name = input("Enter the model name to load (e.g., decision_tree, knn, svm, ann): ")
        load_and_predict(model_name)
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()
