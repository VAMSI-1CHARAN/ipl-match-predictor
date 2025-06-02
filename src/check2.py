import joblib

# After training the model
joblib.dump(model, '../models/match_winner_model.pkl')
print("✅ Model saved successfully!")

