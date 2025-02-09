from flask import Flask

# initialise the Flask app
app = Flask(__name__)

# /hello
@app.route(rule="/hello", methods=["GET"])
def foo():
    return "<H3> Hello User welcome to MLOps class"

if __name__ == "__main__":
    app.run()

# Now go to http://127.0.0.1:5000/hello to se the above return message