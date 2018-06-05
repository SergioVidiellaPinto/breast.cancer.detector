#!/usr/bin/env python3

import sys
import numpy as np
from flask import Flask
from flask import jsonify
from flask import request
from flask import abort

import src.inputanalyzer
from src.classifiersvm import ClassifierSVM

app = Flask(__name__)
classifier = ClassifierSVM()


@app.route('/api/v1.0/analyze', methods=['GET'])
def update_submissions():
    result = inputanalyzer.analyze_dataset()
    return jsonify(result)


@app.route('/api/v1.0/classify', methods=['POST'])
def classify_input():
    if not request.json or not 'inputs' in request.json:
            abort(400)
        
    pred = classifier.make_prediction(np.array(request.json['inputs']))
    prediction = {
        'result': pred
    }
    return jsonify({'task': prediction})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port='80')