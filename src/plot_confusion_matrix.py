import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv('data/matches_processed.csv')

# Define features and target
features = ['venue_encoded', 'batting_first']
target = 'winner'

# One-hot encoding
data_encoded = pd.get_dummies(data[features + [target]], drop_first=True)

# Split
X = data_encoded.drop(columns=[col for col in data_encoded.columns if col.startswith('winner_')])
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load trained model
model = joblib.load('../models/match_winner_model.pkl')

# save model here
joblib.dump(model, '../models/match_winner_model.pkl')
print("✅ Model saved successfully!")

# Predict
y_pred = model.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=sorted(data['winner'].unique()))

# Plot
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=sorted(data['winner'].unique()),
            yticklabels=sorted(data['winner'].unique()))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Match Winner Prediction")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
