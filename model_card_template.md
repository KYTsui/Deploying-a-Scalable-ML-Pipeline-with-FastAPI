# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This is a logistic regression model. Grid-search was performed on the regularization penalty and C hyperparameters. 
Training was conducted using scikit-learn version 1.0.2.

## Intended Use
The model aims to predict whether an individual's annual salary exceeds $50k or not, providing binary classification predictions based on various attributes.

## Training Data
The census income dataset is acquired from the UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/20/census+income
Originally from  
Kohavi,Ron. (1996). Census Income. UCI Machine Learning Repository. https://doi.org/10.24432/C5GP7S
The dataset was split into training and test sets using scikit-learn version 1.0.2, with the training set comprising 80% of the entire dataset.

## Evaluation Data
The test set constitutes 20% of the entire dataset.

## Metrics
The classification model was evaluated using precision, recall, and F1 score, yielding metrics of 0.75, 0.61, and 0.67, respectively.

## Ethical Considerations
Based on the data slicing results (refer to slice_output.txt), bias is evident at the supervised level. 
For example, the model exhibits low metrics when predicting individuals with education levels below bachelor's degrees.

## Caveats and Recommendations
Note that there are some "?" values present in the "native-country" column of the dataset. 
These instances should either be removed or clarified in future versions. 
In addition, exploring correlated features could potentially enhance the performance of future versions of this classification model. 
For example, features such as "age" and "education-num" may be correlated.


