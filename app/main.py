#!/usr/bin/env python3

import numpy as np
from flask import Flask
from flask import jsonify
from flask import request
from flask import abort

import src.inputanalyzer as inputanalyzer
from src.classifiersvm import ClassifierSVM

app = Flask(__name__)
classifier = ClassifierSVM()


@app.route('/api/v1.0/analyze', methods=['GET'])
def update_submissions():
    result = inputanalyzer.analyze_dataset()
    return jsonify(result)


@app.route('/api/v1.0/classify', methods=['POST'])
def classify_input():
    request.get_json(force=True)
    if not request.json or not 'inputs' in request.json:
            abort(400)
    
    input_data = np.array(request.json['inputs'])
    input_attributes = input_data[:, 1:10]
    predictions = classifier.make_prediction(input_attributes)
    res = []
    
    for idx, pred in zip(input_data[:, 0], predictions):
        res.append({"id": str(idx), "class": str(pred)})
        
    result = {
        'result': res
    }
    
    return jsonify(result)


@app.route('/api/v1.0/prediction', methods=['POST'])
def predict_class():
    request.get_json(force=True)
    if not request.json or not 'id' in request.json \
    or not 'clump_thickness' in request.json \
    or not 'uniformity_cell_size' in request.json \
    or not 'uniformity_cell_shape' in request.json \
    or not 'marginal_adhesion' in request.json \
    or not 'single_ephithetial_cell_size' in request.json\
    or not 'bare_nuclei' in request.json \
    or not 'bland_chromatin' in request.json \
    or not 'normal_nucleoli' in request.json or not 'mitoses' in request.json:
        abort(400)
    
    input_array = np.array([[request.json['clump_thickness'], 
                             request.json['uniformity_cell_size'], 
                             request.json['uniformity_cell_shape'],
                             request.json['marginal_adhesion'], 
                             request.json['single_ephithetial_cell_size'], 
                             request.json['bare_nuclei'],
                             request.json['bland_chromatin'], 
                             request.json['normal_nucleoli'], 
                             request.json['mitoses']]])
    
    prediction = classifier.make_prediction(input_array)
        
    result = {
        'id': request.json['id'],
        'class': str(prediction[0])
    }
    
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=80)