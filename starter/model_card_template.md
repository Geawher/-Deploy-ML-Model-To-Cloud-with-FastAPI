# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The used model for preidiction is a RandomForest classifier with default hyperparameters using the Sklearn library.
## Intended Use
This model demonstrates the capability to predict if an individual's salary level is higher or lower than 50K/yr by leveraging different features.
## Training Data
The training data is from the UCI Machine Learning Repository. The data set contains 48,842 instances and 14 attributes. It is available at [here](https://archive.ics.uci.edu/ml/datasets/census+income)
## Evaluation Data
For the evaulation data sliced the main dataset into train and test, with a 80/20 split. The test data is used to evaluate the model's performance.
As for the processing of the data, we put in place categorical encoding using onehot encoding and binazier was used for the target variable.
## Metrics
The model achieved the following scores:
```
Precision: 0.75
Recall: 0.63
Fbeta: 0.68
## Ethical Considerations
Model performance should be treated and analyzed with caution as the training features encompass sensitive information like race and sex. This inclusion may introduce biases in predictions that warrant careful consideration.
## Caveats and Recommendations
The intended use of the model is solely for predicting income based on census data, and it is not designed for any other purposes.