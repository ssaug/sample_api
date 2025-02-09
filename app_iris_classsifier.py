from flask import Flask, request, jsonify
import pickle
import numpy as np
#from sklearn.tree import DecisionTreeClassifier

# Initialise the Flask app
app = Flask(__name__)

# Load the trained classifier model
with open("./model/iris_classifier.pkl", "rb") as f:
    clf = pickle.load(f)

# /get_status
@app.route(rule="/get_status", methods=["GET"])
def foo():
    return {"training": 70, "testing": 30}

# /prediction
@app.route(rule="/prediction", methods=["POST"])
def prediction(): ## This method not possible to open in any browsers , hence will use postman tool for this
    payload = request.json
    #print(payload)
    #return None
    X_unknown = [payload["sepal-length"], payload["sepal-width"], payload["petal-length"], payload["petal-width"]]
    X_unknown = np.array(X_unknown).reshape(1, -1)
    prediction = clf.predict(X_unknown)
    return jsonify({"predicted_value": prediction[0]})


if __name__ == "__main__":
    app.run(port=5003)
