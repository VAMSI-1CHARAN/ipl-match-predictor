import pandas as pd
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def preprocess_input(user_input_dict, feature_columns):
    # Convert input to DataFrame
    input_df = pd.DataFrame([user_input_dict])
    
    # One-hot encode
    input_encoded = pd.get_dummies(input_df)
    
    # Match training columns, fill missing with 0
    input_aligned = input_encoded.reindex(columns=feature_columns, fill_value=0)
    
    return input_aligned

def load_data():
    return pd.read_csv('data/matches_processed.csv')

def train_model(data):
    # Basic features (customize later)
    features = ['team1', 'team2', 'toss_winner', 'venue', 'toss_decision']
    target = 'winner'

    # Encode categorical variables
    data_encoded = pd.get_dummies(data[features + [target]], drop_first=True)

    X = data_encoded.drop(columns=[col for col in data_encoded.columns if 'match_winner_' in col])
    y = data_encoded[[col for col in data_encoded.columns if 'winner_' in col]].idxmax(axis=1).str.replace('winner_', '')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("✅ Accuracy:", accuracy_score(y_test, y_pred))
    print("🧩 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Save model
    joblib.dump(model, '../models/match_winner_model.pkl')
    print("Model saved at ../models/match_winner_model.pkl")
    
    # Save feature columns used
    feature_columns = X.columns.tolist()
    with open('../models/feature_columns.json', 'w') as f:
        json.dump(feature_columns, f) 
    
    return model


if __name__ == "__main__":
    data = load_data()
    model = train_model(data)

