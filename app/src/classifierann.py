import os
from keras.models import model_from_json

import src.config as config


class ClassifierANN():
    
    def __init__(self, classifier=None):
        if classifier:
            self.classifier = classifier
        else:
            self.load_model()
            self.load_weights()

    
    def save_model(self):
        """Save the model as a JSON"""
       
        model_json = self.classifier.to_json()
        file_path = "{}{}".format(os.path.dirname(os.path.realpath(__file__)),
                     config.get_value("paths", "model_nn"))
        with open(file_path, "w") as json_file:
            json_file.write(model_json)


    def save_weights(self):
        """Save the weights of the model as an HDF5 file"""
        
        file_path = "{}{}".format(os.path.dirname(os.path.realpath(__file__)),
                     config.get_value("paths", "weights_nn"))
        self.classifier.save_weights(file_path)
        
    
    def load_model(self):
        """Load json and create model"""
        
        file_path = "{}{}".format(os.path.dirname(os.path.realpath(__file__)),
                     config.get_value("paths", "model_nn"))
        json_file = open(file_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.classifier = model_from_json(loaded_model_json)
        
        
    def load_weights(self):
        """Load weights into new model"""
        
        file_path = "{}{}".format(os.path.dirname(os.path.realpath(__file__)),
                    config.get_value("paths", "weights_nn"))
        self.classifier.load_weights(file_path)
    
    
    def make_prediction(self, input_values, th=0.5):
        """Given an input fitting the model, make a prediction of its class"""
        
        prediction = self.classifier.predict(input_values)
        prediction = (prediction > 0.5)
        
        return prediction