import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    model = Pipeline(
        steps=[("classifier", RandomForestClassifier())]
    )
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)

def compute_slices(dataframe, feature, y, preds):
    """
    Compute the performance on slices for a given categorical feature.

    Inputs:
    Dataframe: pandas DataFrame
        Test dataframe pre-processed with features, including the categorical feature for slicing
        then save the dataframe into a text file appending each time for each feature.
    feature: str
        Feature on which to perform the slices.
    y : np.array
        Corresponding known labels, binarized.
    preds : np.array
        Predicted labels, binarized.

    Returns:
    Dataframe with columns:
        feature value: value of the categorical feature
        n_samples: number of data samples in the slice
        precision : precision score
        recall : recall score
        fbeta : fbeta score
    """
    slice_options = dataframe[feature].unique()
    performance_data = []
    save_path = './starter/starter/ml/slice_output.txt'

    for option in slice_options:
        slice_mask = dataframe[feature] == option
        slice_y = y[slice_mask]
        slice_preds = preds[slice_mask]

        precision = precision_score(slice_y, slice_preds)
        recall = recall_score(slice_y, slice_preds)
        fbeta = fbeta_score(slice_y, slice_preds, beta=1)

        performance_data.append({
            'feature value': option,
            'n_samples': len(slice_y),
            'precision': precision,
            'recall': recall,
            'fbeta': fbeta
        })

    performance_df = pd.DataFrame(performance_data)
    performance_df.to_csv(save_path, mode='a', index=False)
