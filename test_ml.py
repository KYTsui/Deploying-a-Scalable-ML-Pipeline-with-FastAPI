import pytest
import os
from ml.data import process_data
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

@pytest.fixture(scope="function")
def data():
    """ Simple function to generate some fake Pandas data."""
    project_path = ""
    data_path = os.path.join(project_path, "data", "census.csv")
    print(data_path)
    data = pd.read_csv(data_path)

    return data

# First unit test:
def test_no_null(data):
    """
    # Check if the dataset has any null values. This test will pass if there is no null.
    """
    assert data.shape == data.dropna().shape, "Dropping null changes shape of the dataset."


# Second unit test:
def test_binary_classes(data):
    """
    # Check if the label has only 2 classes. If not, this test will fail, likely due to formatting errors.
    """
    unique_classes = data["salary"].unique()
    assert len(unique_classes) == 2, "The label should have only two unique classes (i.e., <=50K and >50K). Check for formatting errors."



# Third unit test:
def test_precision(data):
    """
    # Check if the calculated precision is acceptable (i.e., > 0.65).
    There are reasons for this test to fail, such as having mislabelled data points.
    """
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    # Split data into train and test sets.
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=True,

    )

    X_test, y_test, encoder, lb = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    # Use the train_model function to train the model on the training dataset.
    model = train_model(X_train, y_train)

    # Use the inference function for predictions.
    preds = inference(model, X_test)

    # Use the compute_model_metrics to calculate metrics.
    p, r, fb = compute_model_metrics(y_test, preds)
    expected_p = 0.65
    assert p > expected_p, "Precision value is too low. Check if there are data points being mislabeled."

