# -*- coding: utf-8 -*-

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense

from classifierann import ClassifierANN
import inputanalyzer


def build_classifier(optimizer, units):
    """Build a parametrizable classifier with two Hidden Layers"""
    
    classifier = Sequential()
    
    # Add the Input Layer
    classifier.add(Dense(units = units, kernel_initializer = 'uniform', 
                         activation = 'relu', input_dim = 9))
    
    # Add a Hidden Layer
    classifier.add(Dense(units = units, kernel_initializer = 'uniform', 
                         activation = 'relu'))
    
    # Add the final Layer with sigmoid activation
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', 
                         activation = 'sigmoid'))
    
    # Compile optimizing accuracy
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', 
                       metrics = ['accuracy'])
    
    return classifier


def hyper_parametrization(X_train, y_train):
    """Given a set of parameters, choose the ones optimizing the score"""
    
    classifier = KerasClassifier(build_fn = build_classifier)
    parameters = {'batch_size': [10, 50],
                  'epochs': [100, 500],
                  'optimizer': ['adam', 'rmsprop'],
                  'units': [10, 50]}
    
    grid_search = GridSearchCV(estimator = classifier,
                               param_grid = parameters,
                               scoring = 'accuracy',
                               cv = 10)
    
    
    grid_search = grid_search.fit(X_train, y_train)
    best_parameters = grid_search.best_params_
    
    return best_parameters


def train():
    """Train a new model based in the best possible parameters"""
    
    X_train, X_test, y_train, y_test = inputanalyzer.prepare_input(binarize_input=True, normalize=True)
    
    # Get best parameteres
    best_parameters = hyper_parametrization(X_train, y_train)
    
    # Create the ANN and train it with the obtained best parameters
    classifier = build_classifier(best_parameters['optimizer'], 
                                  best_parameters['units'], 
                                  best_parameters['loss'])
    classifier.fit(X_train, y_train, batch_size=best_parameters['batch_size'],
                   epochs=best_parameters['epochs'])
    
    # Save the model and weights
    resultClassifier = ClassifierANN(classifier=classifier)
    resultClassifier.save_model()
    resultClassifier.save_weights()
    
    # Make a prediction and return the Confusion Matrix, so it can be analyzed
    y_pred = resultClassifier.make_prediction(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    print(best_parameters)
    print(cm)
    

if __name__ == "__main__":
    train()
