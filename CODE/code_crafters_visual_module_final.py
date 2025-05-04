
# Code Crafters ML Application with Visualizations
import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

def load_data():
    base_path = Path(__file__).resolve().parent
    input_train_path = base_path / "INPUT" / "TRAIN" / "fraud_dataset_train.xlsx"
    input_test_path = base_path / "INPUT" / "TEST" / "fraud_dataset_test.xlsx"
    train_df = pd.read_excel(input_train_path)
    test_df = pd.read_excel(input_test_path)
    return train_df, test_df, base_path

def plot_confusion_matrix(y_true, y_pred, model_name, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Fraud", "Fraud"], yticklabels=["Not Fraud", "Fraud"])
    plt.title(f"{model_name.upper()} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_confusion.png")
    plt.close()

def plot_roc_curve(y_true, y_prob, model_name, output_dir):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f"{model_name.upper()} ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_roc.png")
    plt.close()

def save_results(model_name, model, X_test, y_test, test_df, test_preds, base_path, use_proba=False):
    model_path = base_path / "MODEL" / f"{model_name}_model.joblib"
    output_dir = base_path / "OUTPUT"
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)
    test_df["Prediction"] = test_preds
    test_df.to_csv(output_dir / f"{model_name}_predictions.csv", index=False)

    print("Test Accuracy:", accuracy_score(y_test, test_preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, test_preds))
    print("Classification Report:\n", classification_report(y_test, test_preds))

    plot_confusion_matrix(y_test, test_preds, model_name, output_dir)

    if use_proba and hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
        plot_roc_curve(y_test, probs, model_name, output_dir)

def run_model(model_name, model_class, use_proba=False, **kwargs):
    train_df, test_df, base_path = load_data()
    features = ["Amount", "Account_age"]
    target = "Fraud_Label"

    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]

    model = model_class(**kwargs)
    model.fit(X_train, y_train)
    test_preds = model.predict(X_test)
    save_results(model_name, model, X_test, y_test, test_df, test_preds, base_path, use_proba)

def load_and_predict(model_name):
    base_path = Path(__file__).resolve().parent
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
        run_model("decision_tree", DecisionTreeClassifier, use_proba=False, random_state=42)
    elif choice == '2':
        run_model("knn", KNeighborsClassifier, use_proba=False, n_neighbors=5)
    elif choice == '3':
        run_model("svm", SVC, use_proba=True, kernel='rbf', probability=True)
    elif choice == '4':
        run_model("ann", MLPClassifier, use_proba=True, hidden_layer_sizes=(10, 5), max_iter=500, random_state=42)
    elif choice == '5':
        model_name = input("Enter the model name to load (e.g., decision_tree, knn, svm, ann): ")
        load_and_predict(model_name)
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()
