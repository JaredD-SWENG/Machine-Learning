# import Flask
from flask import Flask, send_from_directory, request, json
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
# import ml.py
import ml

# Send index.html
@app.route('/', methods=["GET"])
@app.route('/index.html', methods=["GET"])
def get_index():
    #return contents of index.html
    return send_from_directory('', 'index.html', mimetype='text/html')

# Send main.js
@app.route('/main.js', methods=["GET"])
def get_main():
     #return contents of main.js
    return send_from_directory('', 'main.js', mimetype='text/javascript')

# Send the result from machine learning
# Endpoint is "result"
@app.route('/result', methods=["GET"])
def result():

    # call the prediction function in ml.py
    results = ml.prediction()
    
    # create a dictionary with all three results
    resultDict = { 
        "model": "kNN",
        "accuracy": results[0],
        "precision": results[1],
        "recall": results[2]
    }
    
    # convert dictionary to JSON string
    resultString = json.dumps(resultDict)

    return resultString

# Run the server
if __name__ == '__main__':
    
    # train the model
    ml.train()
        
    # start the server
    app.run(port = 8000)