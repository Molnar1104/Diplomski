import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit

def train_and_evaluate_model():
    """
    Loads the dataset, trains a Random Forest model, and evaluates its performance.
    """
    # --- 1. Load Data ---
    dataset_path = 'final_dataset.csv'
    try:
        data = pd.read_csv(dataset_path, index_col='Date', parse_dates=True)
        print("Dataset loaded successfully.")
    except FileNotFoundError:
        print(f"ERROR: The dataset file was not found at {dataset_path}")
        print("Please run data_collector.py first to generate the dataset.")
        return

    # --- 2. Prepare Data ---
    # Define features (X) and target (y)
    # The target is whether the next day's close is higher (1) or not (0)
    features = [col for col in data.columns if col != 'Target_Next_Day']
    X = data[features]
    y = data['Target_Next_Day']

    # --- 3. Time-Series Split ---
    # Use a TimeSeriesSplit to ensure we train on past data and test on future data
    # This prevents "lookahead bias"
    tscv = TimeSeriesSplit(n_splits=5)

    # We will use the last split for a simple train/test evaluation
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")

    # --- 4. Train the Model ---
    print("\nTraining the Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("Model training complete.")

    # --- 5. Evaluate the Model ---
    print("\nEvaluating the model on the test set...")
    y_pred = model.predict(X_test)

    # Calculate and print metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    print("\nConfusion Matrix:")
    print(conf_matrix)

if __name__ == '__main__':
    train_and_evaluate_model()
