import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def main():
    print("Loading dataset...")
    df = pd.read_csv('loan_approval_dataset.csv')
    
    print("Cleaning data...")
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    # Strip whitespace from string columns
    string_cols = df.select_dtypes(include=['object']).columns
    for col in string_cols:
        df[col] = df[col].str.strip()
        
    # Drop loan_id
    if 'loan_id' in df.columns:
        df = df.drop('loan_id', axis=1)
        
    print("Encoding categorical variables...")
    # Encoding dictionaries
    edu_map = {'Graduate': 1, 'Not Graduate': 0}
    emp_map = {'Yes': 1, 'No': 0}
    status_map = {'Approved': 1, 'Rejected': 0}
    
    df['education'] = df['education'].map(edu_map)
    df['self_employed'] = df['self_employed'].map(emp_map)
    df['loan_status'] = df['loan_status'].map(status_map)
    
    # Check for NaNs
    df = df.dropna()
    
    X = df.drop('loan_status', axis=1)
    y = df['loan_status']
    
    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Saving model and metadata...")
    model_data = {
        'model': model,
        'features': list(X.columns),
        'edu_map': edu_map,
        'emp_map': emp_map
    }
    joblib.dump(model_data, 'model.pkl')
    print("Model saved to model.pkl successfully.")

if __name__ == '__main__':
    main()
