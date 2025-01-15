import numpy as np
import os
import pandas as pd

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from ucimlrepo import fetch_ucirepo


def load_uci_dataset(file_path, repo_id=None, verbose=False):
    """
    Load a dataset from the UCI Machine Learning Repository.

    Parameters
    ----------
    file_path : str
        The path to the dataset file.
    repo_id : int, optional
        The repository ID to download the dataset if the file does not exist.
    verbose : bool, optional
        If True, print metadata of the dataset.

    Returns
    -------
    pd.DataFrame
        The loaded dataset.

    Raises
    ------
    ValueError
        If the file does not exist and no repo_id is provided.
    """
    if verbose:
        if repo_id is None:
            raise ValueError("Repo ID is required to show metadata of the dataset!")
        dataset = fetch_ucirepo(id=repo_id)
        print(dataset.metadata)

    if not os.path.exists(file_path):
        if repo_id is None:
            raise ValueError("Repo ID is required to download the dataset! It doesn't exist by default.")

        dataset = fetch_ucirepo(id=repo_id)
        features = dataset.data.features
        targets = dataset.data.targets.squeeze()

        combined_df = pd.concat([features, targets], axis=1)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        combined_df.to_csv(file_path, index=False)
    else:
        dataset = pd.read_csv(file_path)

    return dataset


def train_ensemble_models(x_train, x_test, y_train, y_test, probabilities,
                          n_ensembles=5, n_features_sample=5, random_state=42, verbose=False):
    """
    Train multiple ensemble models on probabilistically sampled subsets of features and evaluate their performance.

    Parameters
    ----------
    x_train : pd.DataFrame
        Training data containing features. Each row is a sample, and each column is a feature.
    x_test : pd.DataFrame
        Test data containing features. Each row is a sample, and each column is a feature.
    y_train : pd.Series
        Training labels corresponding to `x_train`. Each entry is the label for the corresponding row in `x_train`.
    y_test : pd.Series
        Test labels corresponding to `x_test`. Each entry is the label for the corresponding row in `x_test`.
    probabilities : np.ndarray or list
        Array of probabilities for selecting each feature. This array is used to probabilistically sample subsets of features
        for each ensemble model. Length of `probabilities` should match the number of features in `x_train`.
    n_ensembles : int, optional, default=5
        The number of ensemble models to train.
    n_features_sample : int, optional, default=5
        The number of features to sample for each ensemble model.
    random_state : int, optional, default=42
        Random seed for reproducibility of results.
    verbose : bool, optional, default=False
        Whether to print progress information during the training of the ensemble models.

    Returns
    -------
    list of dicts
        A list of dictionaries where each dictionary contains the results for one model in one ensemble, including:
        - "Ensemble": The ensemble number.
        - "Classifier": The name of the classifier.
        - "Accuracy": The accuracy of the classifier on the test data.
        - "Sampled Features": The list of feature indices that were sampled for the ensemble.
    """
    ensemble_results = []

    # Define classifiers
    classifiers = {
        "Random Forest": RandomForestClassifier(random_state=random_state),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
        "AdaBoost": AdaBoostClassifier(random_state=random_state),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "SVM": SVC(kernel='rbf', probability=True, random_state=random_state),
        "LDA": LinearDiscriminantAnalysis()
    }

    # Sample feature subset and train each classifier
    for ensemble in range(n_ensembles):
        if verbose:
            print(f"\nTraining Ensemble {ensemble + 1}/{n_ensembles}...")

        # Sample feature subset
        sampled_features = sample_feature_subset(probabilities, n_features_sample)
        x_train_sampled = x_train.iloc[:, sampled_features]
        x_test_sampled = x_test.iloc[:, sampled_features]

        # Train and evaluate each classifier
        for name, clf in classifiers.items():
            if verbose:
                print(f"Training {name}...")

            clf.fit(x_train_sampled, y_train)
            y_pred = clf.predict(x_test_sampled)
            accuracy = accuracy_score(y_test, y_pred)
            ensemble_results.append({
                "Ensemble": ensemble + 1,
                "Classifier": name,
                "Accuracy": accuracy,
                "Sampled Features": sampled_features
            })

            if verbose:
                print(f"{name} Accuracy: {accuracy:.4f}")

    return ensemble_results


def sample_feature_subset(probs, n_features):
    """
    Sample a subset of features based on given probabilities.

    Parameters
    ----------
    probs : np.ndarray
        Array of probabilities for each feature.
    n_features : int
        Number of features to sample.

    Returns
    -------
    np.ndarray
        Indices of the sampled features.
    """
    feature_indices = np.arange(len(probs))
    sampled_indices = np.random.choice(feature_indices, size=n_features, p=probs, replace=False)

    return sampled_indices


def compute_feature_frequency(ensemble_results, n_features):
    """
    Compute the frequency of each feature being selected across all ensemble models.

    Parameters
    ----------
    ensemble_results : list of dicts
        List of dictionaries containing the results of the ensemble models.
    n_features : int
        Total number of features.

    Returns
    -------
    np.ndarray
        Array of feature frequencies.
    """
    ensemble_results_df = pd.DataFrame(ensemble_results)

    feature_counts = np.zeros(n_features)
    for result in ensemble_results:
        sampled_features = result["Sampled Features"]
        feature_counts[sampled_features] += 1

    feature_frequency = feature_counts / (max(set(ensemble_results_df["Ensemble"])) * len(set(ensemble_results_df["Classifier"])))

    return feature_frequency
