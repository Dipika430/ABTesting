# train_model.py
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

# Model A
model_A = RandomForestClassifier(n_estimators=10)
model_A.fit(X, y)

# Model B (improved)
model_B = RandomForestClassifier(n_estimators=50)
model_B.fit(X, y)

# Save models
pickle.dump(model_A, open("model_A.pkl", "wb"))
pickle.dump(model_B, open("model_B.pkl", "wb"))

print("✅ Both models saved")