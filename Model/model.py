import json
import numpy as np
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# load dataset
with open("Model/sample_users.json","r") as f:
    data = json.load(f)

X = []
y = []

for user in data:

    try:
        features = [
            user["rating"],
            user["maxRating"],
            user["SolvedCount"],
            user["registrationTimeSeconds"]
        ]

        X.append(features)
        y.append(user["maxRating"])

    except:
        continue


X = np.array(X)
y = np.array(y)

# train test split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Random Forest model
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train,y_train)

# evaluate
pred = model.predict(X_test)

error = mean_absolute_error(y_test,pred)

print("Model trained successfully")
print("Mean Absolute Error:",error)

# save model
pickle.dump(model,open("Model/regression_model.pkl","wb"))

print("Model saved as regression_model.pkl")