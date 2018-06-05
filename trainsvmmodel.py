# -*- coding: utf-8 -*-

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn import svm

import inputanalyzer
from classifiersvm import ClassifierSVM


def build_classifier(C, kernel):
    """Build a parametrizable classifier with two Hidden Layers"""
    
    return svm.SVC(C=C, kernel=kernel)


def hyper_parametrization(X_train, y_train):
    """Given a set of parameters, choose the ones optimizing the score"""
    
    classifier = svm.SVC()
    parameters = {'C': [1, 25, 50, 100, 200],
                  'kernel': ["linear", "rbf"]}
    
    grid_search = GridSearchCV(estimator = classifier,
                               param_grid = parameters,
                               scoring = 'accuracy',
                               cv = 10)
    
    grid_search = grid_search.fit(X_train, y_train)
    best_parameters = grid_search.best_params_
    
    return best_parameters


def train():
    """Train a new model based in the best possible parameters"""
    
    X_train, X_test, y_train, y_test = inputanalyzer.prepare_input()
    
    # Get best parameteres
    best_parameters = hyper_parametrization(X_train, y_train)
    
    # Create the ANN and train it with the obtained best parameters
    classifier = build_classifier(best_parameters['C'], 
                                  best_parameters['kernel'])
    classifier.fit(X_train, y_train)
    
    # Save the model and weights
    resultClassifier = ClassifierSVM(classifier=classifier)
    resultClassifier.save_model()
    
    # Make a prediction and return the Confusion Matrix, so it can be analyzed
    y_pred = resultClassifier.make_prediction(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    print(best_parameters)
    print(cm)
    
    y_pred = resultClassifier.make_prediction(X_train)
    cm = confusion_matrix(y_train, y_pred)
    
    print(cm)
    print(resultClassifier.classifier.score(X_train, y_train))
    print(resultClassifier.classifier.score(X_test, y_test))
    

if __name__ == "__main__":
    train()