from sklearn.model_selection import cross_val_score
from keras.models import model_from_json
import os

import inputanalyzer
import config


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
                     config.get_value("paths", "model_path"))
        with open(file_path, "w") as json_file:
            json_file.write(model_json)


    def save_weights(self):
        """Save the weights of the model as an HDF5 file"""
        
        file_path = "{}{}".format(os.path.dirname(os.path.realpath(__file__)),
                     config.get_value("paths", "weights_path"))
        self.classifier.save_weights(file_path)
        
    
    def load_model(self):
        """Load json and create model"""
        
        json_file = open(config.get_value("paths", "model_path"), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.classifier = model_from_json(loaded_model_json)
        
        
    def load_weights(self):
        """Load weights into new model"""
        
        self.classifier.load_weights(config.get_value("paths", "weights_path"))
    
    
    def crossvalidate_model(self, cv=10):
        """Make a crossvalidation over the model and obtain the mean accuracy
        and its variance"""
        
        X_train, X_test, y_train, y_test = inputanalyzer.prepare_input()
        accuracies = cross_val_score(estimator = self.classifier,
                                     X = X_train, y = y_train, cv = cv)
        mean = accuracies.mean()
        variance = accuracies.std()
        
        return mean, variance
    
    
    def make_prediction(self, input_values, th=0.5):
        """Given an input fitting the model, make a prediction of its class"""
        
        prediction = self.classifier.predict(input_values)
        prediction = (prediction > 0.5)
        
        return prediction
    

if __name__ == '__main__':
    c = ClassifierANN()
    c.load_model()
    c.load_weights()
    c.classifier.score