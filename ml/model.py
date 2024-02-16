import pickle
from sklearn.metrics import fbeta_score, precision_score, recall_score
from ml.data import process_data
# Note that logistic regression will be used for this classification task.
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

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
    # Define the model.
    model = LogisticRegression(max_iter=1000)

    # Define hyperparameters for tuning regularization penalty and other parameters.
    param_grid = {
        'penalty': ['None', 'l2'],
        'C': [0.01, 0.1, 1, 10, 100] }

    # Initialize GridSearchCV.
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1)

    # Perform hyperparameter tuning.
    grid_search.fit(X_train, y_train)

    # Get the best model from hyperparameter tuning.
    model = grid_search.best_estimator_

    # Train the best model.
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
    model : LogisticRegression()
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    # Use the trained model to make predictions only.
    preds = model.predict(X)

    return preds

def save_model(model, path):
    """ Serializes model to a file.

    Inputs
    ------
    model
        Trained machine learning model or OneHotEncoder.
    path : str
        Path to save pickle file.
    """
    with open(path, 'wb') as f:
        pickle.dump(model, f)

def load_model(path):
    """ Loads pickle file from `path` and returns it."""
    try:
        with open(path, 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        print("Error: File not found at the specified path.")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    return model


def performance_on_categorical_slice(
    data, column_name, slice_value, categorical_features, label, encoder, lb, model
):
    """ Computes the model metrics on a slice of the data specified by a column name and

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Inputs
    ------
    data : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    column_name : str
        Column containing the sliced feature.
    slice_value : str, int, float
        Value of the slice feature.
    categorical_features: list
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.
    model : LogisticRegression()
        Model used for the task.

    Returns
    -------
    precision : float
    recall : float
    fbeta : float

    """
    # TODO: implement the function
    X_slice, y_slice, encoder, lb = process_data(
        # your code here
        # for input data, use data in column given as "column_name", with the slice_value 
        # use training = False
        data[data[column_name] == slice_value],
        categorical_features=categorical_features,
        label=label,
        training=False,
        encoder=encoder,
        lb=lb
    )

    preds = inference(model, X_slice)
    precision, recall, fbeta = compute_model_metrics(y_slice, preds)
    return precision, recall, fbeta
