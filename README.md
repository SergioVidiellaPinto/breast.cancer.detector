# breast.cancer.detector

The aim of this project is to provide a classifier ready for an end-user
capable of classify a set of data into a resultant benign or malignant tumour.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.
Instructions and deployment have been tested in a Centos7 machine.

### Prerequisites

* Python 3.6
* Flask 0.11.1
* keras 2.1.6
* scikit-learn 0.19.1
* pandas 0.23.0

### Installing

If you want to test the API Rest in your local, Deployment step is highly recommended.

For working in local or retrain the system, just download the code.
```
git clone https://github.com/SergioVidiellaPinto/breast.cancer.detector.git
```

If you want to run the API in your local host
```
python app/main.py
```

If you want to retrain the ANN (Artificial Neural Network) and see some results
```
python app/src/trainnnmodel.py
```

If you want to retrain the SVM (Support Vector Machine) and see some results
```
python app/src/trainsvmmodel.py
```

### API REST

The methods of the API are the following
* **/api/v1.0/analyze** [GET]

Expected result is
```
{
    'column_1': {
        'value_1': 145,
        'value_2': 69,
        ...
        'value_nan': 0
    },
    ...
    'column_10': {
        'value_1': 145,
        'value_2': 69,
        ...
        'value_nan': 0
    }     
}
```

* **/api/v1.0/classify** [POST]

Expected input is
```
{
    'inputs': [[2554, 1, 2, 1, 1, 1, 2, 4, 1, 2], ..., [2557, 5, 2, 4, 7, 8, 10, 10, 9, 7]]
}
```

Expected result is
```
{
    'result': [
        {
            'id': 2554,
            'class': 2
        },
        ...
        {
            'id': 2557,
            'class': 4
        }
    ]
}
```

* **/api/v1.0/prediction** [POST]

Expected input is
```
{
    'id': 2554, 
    'clump_thickness': 1,
    'uniformity_cell_size': 2,
    'uniformity_cell_shape': 1,
    'marginal_adhesion': 1,
    'single_ephithetial_cell_size': 1,
    'bare_nuclei': 2,
    'bland_chromatin': 4,
    'normal_nucleoli': 1, 
    'mitoses': 2
}
```

Expected result is
```
{
    'id': 2554,
    'class': 2
}
```

## Deployment

For deployment it is recommended to do it with Docker, so Docker is required to be installed.
Custom deployments are also possible.

First clone the repository
```
git clone https://github.com/SergioVidiellaPinto/breast.cancer.detector.git
```

Then, build the Docker container. Name is a custom name you want to put to your local image. 
PATH_TO_CODE is the path of the folder where the code was coded.
```
docker build -t {NAME} {PATH_TO_CODE}
```

Once build the image, just run it and it will automatically will be deployed 
and attached to one port on your localhost. Make sure {DST_PORT} is free.
```
docker run -p {PORT}:80 -t {NAME}
```


## Authors

* **Sergio Vidiella**