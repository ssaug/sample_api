from flask import Flask

# initialise the Flask app
app = Flask(__name__)

# <TO-DO> create a classifier and return the training data shape
"""Classifier code"""

# /get_status
@app.route('/get_status', methods=["GET"])
def foo():
    return {"training": 70, "testing": 30}

if __name__ == "__main__":
    app.run(port=5003)
