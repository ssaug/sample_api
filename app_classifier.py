#< To:Do> Assignment 1:  create a classifier & return training data shape
"""Classifier code to give testing and training data"""

from flask import Flask
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Initialize Flask app
app = Flask(__name__)

# Load dataset
df = pd.read_csv("titanic.csv")

# Preprocess dataset
df = df.fillna(0)  # Fill missing values
df["gender_enc"] = df["Sex"].astype('category').cat.codes
df["embark_enc"] = df["Embarked"].astype('category').cat.codes

# Select features and target
X = df[["Pclass", "Age", "gender_enc", "embark_enc", "Fare", "SibSp", "Parch"]]
Y = df["Survived"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=42)

# Train Decision Tree Classifier
model = DecisionTreeClassifier(criterion="entropy", max_depth=5)
model.fit(X_train, y_train)

# API Endpoint to get training data shape
@app.route("/get_status", methods=["GET"])
def get_status():
    return {
        "training_data_shape": list(X_train.shape),  # Convert tuple to list
        "testing_data_shape": list(X_test.shape)     # Convert tuple to list
    }

### output : {"testing_data_shape":[357,7],"training_data_shape":[534,7]}

# Run the Flask app
if __name__ == "__main__":
    app.run(port=5000)

# Now go to http://127.0.0.1:5000/hello to se the above return message