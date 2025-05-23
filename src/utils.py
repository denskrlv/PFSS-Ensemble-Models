import numpy as np
import pandas as pd

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from tqdm import tqdm
from ucimlrepo import fetch_ucirepo


def load_uci_dataset(repo_id):
    """
    Load a dataset from the UCI Machine Learning Repository.

    Parameters
    ----------
    repo_id : int
        The repository ID to download the dataset.

    Returns
    -------
    pd.DataFrame
        The loaded dataset.

    Raises
    ------
    ValueError
        If the file does not exist and no repo_id is provided.
    """
    if repo_id is None:
        raise ValueError("Repo ID is required to download the dataset!")

    dataset = fetch_ucirepo(id=repo_id)
    features = dataset.data.features
    targets = dataset.data.targets.squeeze()
    combined_df = pd.concat([features, targets], axis=1)

    return dataset.metadata, combined_df


def normalize_dataset(dataset, num_bins):
    continuous_vars = dataset.select_dtypes(include=['float', 'int']).columns
    print("Continuous columns identified:", continuous_vars)
    for col in continuous_vars:
        dataset[col] = pd.cut(dataset[col], bins=num_bins, labels=[f'Bin {i+1}' for i in range(num_bins)])
    return dataset


def train_ensemble_models(x_train, x_test, y_train, y_test, probabilities=None, cv=5,
                          n_ensembles=5, n_features_sample=5, random_state=42, verbose=False):
    """
    Train multiple ensemble models on probabilistically sampled subsets of features,
    perform GridSearchCV for hyperparameter tuning for each model, and evaluate their performance.

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
    cv: int, optional, default=5
        Number of cross-validation folds for GridSearchCV.
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
        - "Best Params": Best hyperparameters found by GridSearchCV.
    """
    ensemble_results = []

    # Define classifiers and their hyperparameter grids
    classifiers = {
        "Random Forest": (RandomForestClassifier(random_state=random_state),
                          {'n_estimators': [50, 100], 'max_depth': [10, 20, None]}),
        "Gradient Boosting": (GradientBoostingClassifier(random_state=random_state),
                              {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1, 1.0]}),
        "AdaBoost": (AdaBoostClassifier(random_state=random_state),
                     {'n_estimators': [50, 100], 'learning_rate': [0.5, 1.0, 2.0]}),
        "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=random_state),
                               {'C': [0.1, 1.0, 10.0], 'penalty': ['l2']}),
        "SVM": (SVC(kernel='rbf', probability=True, random_state=random_state),
                {'C': [1, 10, 100], 'gamma': ['scale', 'auto']}),
        "LDA": (LinearDiscriminantAnalysis(),
                {'solver': ['lsqr', 'eigen'], 'shrinkage': [None, 'auto']}),
    }

    # Sample feature subset and train each classifier
    for ensemble in tqdm(range(n_ensembles), desc="Ensemble Models with GridSearch", leave=False):
        # Check that the probabilities array is not empty, otherwise train on all features
        if probabilities is None:
            x_train_sampled = x_train
            x_test_sampled = x_test
            sampled_features = np.arange(x_train.shape[1])
        else:
            sampled_features = sample_feature_subset(probabilities, n_features_sample)
            x_train_sampled = x_train.iloc[:, sampled_features]
            x_test_sampled = x_test.iloc[:, sampled_features]

        # Train and evaluate each classifier with GridSearchCV
        for name, (clf, param_grid) in classifiers.items():
            grid_search = GridSearchCV(clf, param_grid, cv=cv, verbose=verbose, n_jobs=-1)
            grid_search.fit(x_train_sampled, y_train)
            best_clf = grid_search.best_estimator_
            best_params = grid_search.best_params_

            # Make predictions with the best estimator
            y_pred = best_clf.predict(x_test_sampled)
            accuracy = accuracy_score(y_test, y_pred)

            ensemble_results.append({
                "Ensemble": ensemble + 1,
                "Classifier": name,
                "Accuracy": accuracy,
                "Sampled Features": sampled_features,
                "Best Params": best_params
            })

            if verbose:
                print(f"{name} - Best Params: {best_params}")
                print(f"{name} - Accuracy: {accuracy:.4f}")

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
