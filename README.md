# Keras Farsi Digits

Training deep learning models with Persian and English digits data set using keras.

## Project description

Implementing deep CNN models using keras to perform classification task on 
Persian digits data set ( aka Hoda ) and English digits data set ( MNIST ).

First - Training the model to classify Farsi digits and using the same trained model
( with all weights, parameters and layers structure ) for the English data set.

Second - Using saved weights from previous training and tuning final layers 
to improve English data set classification accuracy.  
 
## Prerequisites

Required python libraries 

1- keras 2- numpy 3- opencv-python

you can use commands like

```
pip install keras

pip install numpy

pip install opencv-python

```

## Files description 

1- DigitDB : Containing Farsi digits data set train and test .cdb files.

2- HodaDatasetReader.py : python code to read digits images and convert to training
numeric values.

3- main.py : Implementing models using python. Code descriptions are provided inside the file. 



