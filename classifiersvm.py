import os
import pickle

import config


class ClassifierSVM():
    
    def __init__(self, classifier=None):
        if classifier:
            self.classifier = classifier
        else:
            self.load_model()

    
    def save_model(self):
        """Save the model as a JSON"""
       
        file_path = "{}{}".format(os.path.dirname(os.path.realpath(__file__)),
                     config.get_value("paths", "model_svm"))
        pickle.dump(self.classifier, open(file_path, 'wb'))
        
    
    def load_model(self):
        """Load json and create model"""
        
        file_path = "{}{}".format(os.path.dirname(os.path.realpath(__file__)),
                     config.get_value("paths", "model_svm"))
        self.classifier = pickle.load(open(file_path, 'rb'))
        
    
    def make_prediction(self, input_values, th=0.5):
        """Given an input fitting the model, make a prediction of its class"""
        
        prediction = self.classifier.predict(input_values)
        
        return prediction