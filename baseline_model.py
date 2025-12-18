import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit

def train_and_evaluate_model():
    # --- 1. Load Data ---
    dataset_path = 'final_dataset.csv'
    try:
        data = pd.read_csv(dataset_path, index_col='Date', parse_dates=True)
        print("Dataset loaded successfully.")
    except FileNotFoundError:
        print(f"ERROR: {dataset_path} not found.")
        return

    # --- 2. Prepare Data ---
    features = [col for col in data.columns if col != 'Target_Next_Day']
    X = data[features]
    y = data['Target_Next_Day']

    # --- 3. Split & Train ---
    tscv = TimeSeriesSplit(n_splits=5)
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    print(f"Training set: {len(X_train)} | Testing set: {len(X_test)}")

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # --- 4. Evaluate ---
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # --- 5. VISUALIZE FEATURE IMPORTANCE (Crucial for Thesis) ---
    importances = model.feature_importances_
    feature_names = X.columns
    
    # Create a DataFrame for nice plotting
    feature_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_df = feature_df.sort_values(by='Importance', ascending=False)

    print("\n--- Feature Importance ---")
    print(feature_df)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_df, palette='viridis')
    plt.title('Random Forest Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance_2.png')
    plt.close()
    print("Saved plot to feature_importance_2.png")

if __name__ == '__main__':
    train_and_evaluate_model()