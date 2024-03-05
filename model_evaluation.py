import os
from typing import List, Tuple, Optional, Dict, Any

import hiplot as hip
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, make_scorer
from sklearn.model_selection import KFold, learning_curve, GridSearchCV

ModelListType = List[Tuple[str, BaseEstimator]]


class GridSearchEvaluator:
    """
    A class for evaluating and comparing multiple scikit-learn models using GridSearchCV
    for hyperparameter tuning and visualizing the results with HiPlot.

    Attributes:
        models (List[Tuple[str, Any]]): A list of tuples where each tuple contains a model name and its instance.
        param_grid (Dict[str, Dict[str, List[Any]]]): A dictionary mapping model names to their hyperparameter search space.
        cv_folds (int): The number of cross-validation folds to use for GridSearchCV.
        random_state (Optional[int]): An optional random state for reproducibility.
        results (Dict[str, Dict[str, Any]]): A dictionary to store the results of GridSearchCV for each model.
    """

    def __init__(self,
                 models: ModelListType,
                 param_grid: Dict[str, Dict[str, List[Any]]],
                 cv_folds: int = 10,
                 random_state: Optional[int] = None):
        """
        Initializes the GridSearchEvaluator with models, their parameter grid, and other configurations.

        Args:
            models (List[Tuple[str, Any]]): A list of tuples containing model names and their instances.
            param_grid (Dict[str, Dict[str, List[Any]]]): A dictionary mapping model names to their hyperparameter grids.
            cv_folds (int, optional): The number of folds for cross-validation. Defaults to 10.
            random_state (Optional[int], optional): A seed for random number generation for reproducibility. Defaults to None.
        """
        self._validate_models_with_grid(models, param_grid)
        self.models = models
        self.param_grid = param_grid
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.results = {}

    @staticmethod
    def _validate_models_with_grid(models: ModelListType, param_grid: Dict[str, Dict[str, List[Any]]]):
        """
        Validates that each model has a corresponding parameter grid defined.

        Args:
            models (List[Tuple[str, Any]]): The list of models to validate.
            param_grid (Dict[str, Dict[str, List[Any]]]): The parameter grid to validate against.

        Raises:
            ValueError: If any model does not have a corresponding parameter grid defined.
        """

        missing_grids = [name for name, _ in models if name not in param_grid]
        if missing_grids:
            raise ValueError(f"No parameter grid provided for models: {', '.join(missing_grids)}")

    @staticmethod
    def _quartic_error_score(y_true, y_pred):
        """
        Custom scoring function that penalizes prediction errors by raising them to the power of 4.
        This emphasizes penalizing larger errors more severely than smaller ones, acting as a severe error penalization score.

        Args:
            y_true (array-like): True labels.
            y_pred (array-like): Predicted labels.

        Returns:
            float: The mean of the penalized errors.
        """
        errors = np.abs(y_pred - y_true)
        penalized_error = np.mean(errors ** 4)
        return penalized_error

    def gridsearch_train(self,
                         X: pd.DataFrame,  # noqa
                         y: pd.Series,
                         scoring: str = None,
                         verbose: bool = True,
                         n_jobs: int = -1) -> None:
        """
        Performs GridSearchCV for each model using the provided dataset and hyperparameters.

        Args:
            X (DataFrame): The input features.
            y (Series): The target variable.
            scoring (str, optional): The scoring metric to use for evaluation. Defaults to a quartic error score to penalize outliers.
            verbose (bool, optional): Whether to print verbose messages during training. Defaults to True.
            n_jobs (int, optional): The number of jobs to run in parallel. Defaults to -1 (use all processors).
        """
        if scoring is None:
            scoring = make_scorer(self._quartic_error_score, greater_is_better=False)

        for name, model in self.models:
            print(f"Evaluating {name} with GridSearchCV")
            grid_search = GridSearchCV(estimator=model,
                                       param_grid=self.param_grid[name],
                                       scoring=scoring,
                                       n_jobs=n_jobs,
                                       cv=self.cv_folds,
                                       verbose=int(verbose))
            grid_search.fit(X, y)
            self.results[name] = {
                'best_score': grid_search.best_score_,
                'best_parameters': grid_search.best_params_,
                'cv_results': grid_search.cv_results_
            }

    def hiplot_results(self, model_name: str) -> Optional[str]:
        """
        Visualizes the GridSearchCV results for a specific model using HiPlot.

        Args:
            model_name (str): The name of the model to visualize the results for.

        Returns:
            Optional[str]: A message indicating the status of the visualization process.
        """
        if model_name not in self.param_grid or not self.param_grid[model_name]:
            return f"There were no hyperparameters to tune for {model_name}, so no plot can be generated."

        data_for_hiplot = []
        cv_results = self.results[model_name]['cv_results']
        for i in range(len(cv_results['params'])):
            row = {'model': model_name}
            row.update(cv_results['params'][i])
            row['mean_test_score'] = cv_results['mean_test_score'][i]
            data_for_hiplot.append(row)

        if not data_for_hiplot:
            return f"No results to plot for {model_name}."

        exp = hip.Experiment.from_iterable(data_for_hiplot)
        exp.display()


class ModelEvaluator:
    """
    A comprehensive model evaluation class designed for assessing and comparing the performance of multiple
    machine learning models. It leverages cross-validation to estimate model performance and supports functionalities
    for training models, evaluating them on a validation set, plotting error distributions and learning curves, and
    saving/loading models. This class aims to provide a streamlined workflow for model selection and evaluation.

    Parameters:
    - models (ModelListType): A list of tuples, each containing a string (model name) and a model instance
      (BaseEstimator) that adheres to the scikit-learn estimator interface.
    - cv_folds (int): The number of folds for cross-validation to evaluate model performance. Default is 10.
    - random_state (Optional[int]): An optional random state for reproducibility of results. Default is None.

    Attributes:
    - models (List[Tuple[str, Any]]): Stores the list of model names and their instances.
    - cv_folds (int): Number of cross-validation folds.
    - random_state (Optional[int]): Seed for random number generation..
    - results (Dict): Dictionary to store evaluation metrics for each model.
    - names (List[str]): List of model names for easy reference.
    - errors (Dict): Dictionary to store raw error values for error distribution plots.
    - model_paths (Dict): Dictionary to store file paths of saved models.

    Methods:
    - train_models: Trains each model using K-Fold cross-validation and stores performance metrics.
    - plot_comparison: Plots a comparison of error distributions across all models.
    - evaluate_validation: Evaluates the models on a separate validation dataset.
    - plot_validation_comparison: Plots a comparison of validation error distributions across all models.
    - plot_learning_curves: Generates learning curves for each model to visualize performance over varying data sizes.
    - save_model: Saves a trained model to disk.
    - load_model: Loads a saved model from disk.
    """

    def __init__(self,
                 models: ModelListType,  # noqa
                 cv_folds: int = 10,
                 random_state: Optional[int] = None):
        """
        Initializes the ModelEvaluator class with a set of models, the number of folds for cross-validation,
        and an optional random state for reproducibility.

        Parameters:
            models (ModelListType): A list of tuples, each containing a string (model name) and a model instance (BaseEstimator).
            cv_folds (int): The number of folds for cross-validation. Defaults to 10.
            random_state (Optional[int]): An optional random state for reproducibility. Defaults to None.
        """
        self.models = models
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.results = {}
        self.names = [name for name, _ in models]  # Initialize model names list
        self.errors = {}  # To store raw errors for boxplot
        self.model_paths = {}  # Dictionary to store paths of saved models
        self.detailed_train_results = {}  # To store training details
        self.detailed_validation_results = {}  # To store validation details

    def train_models(self,
                     X: pd.DataFrame,  # noqa
                     y: pd.Series,
                     verbose: bool = False,
                     plot_comparison: bool = False) -> None:
        """
        Evaluates the performance of each model on the provided dataset using K-Fold cross-validation,
        and optionally plots a comparison of their error distributions.

        Parameters:
            X (pd.DataFrame): The input features to train the model.
            y (pd.Series): The target values for the training data.
            verbose (bool): If True, prints summary statistics for each model. Defaults to False.
            plot_comparison (bool): If True, plots the error distributions for each model. Defaults to False.
        """

        for name, model in self.models:
            kfold = KFold(n_splits=self.cv_folds, random_state=self.random_state, shuffle=True)
            fold_errors = []  # To store errors for boxplot
            rmse_values = []
            mape_values = []
            model_results = []

            for train_index, test_index in kfold.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

                errors = predictions - y_test
                fold_errors.extend(errors)  # Store raw errors for boxplot

                fold_results = pd.DataFrame({
                    'ID': X_test.index,
                    'Truth': y_test,
                    'Predicted': predictions,
                    'Error': errors
                })
                model_results.append(fold_results)

                # Calculate RMSE and MAPE for each fold
                rmse_values.append(np.sqrt(mean_squared_error(y_test, predictions)))
                mape_values.append(mean_absolute_percentage_error(y_test, predictions) * 100)

            # Aggregate fold results and store in the class
            concatenated_results = pd.concat(model_results, ignore_index=True)
            self.detailed_train_results[name] = concatenated_results

            # Calculate and store overall RMSE and MAPE
            overall_rmse = np.mean(rmse_values)
            overall_mape = np.mean(mape_values)

            self.results[name] = {
                'RMSE': overall_rmse, 'MAPE (%)': overall_mape
            }
            self.errors[name] = fold_errors  # Store errors for boxplot

            # Print summary statistics
            if verbose:
                msg = (f"{name} - RMSE: {overall_rmse:.2f}, "
                       f"MAPE: {overall_mape:.2f}%")
                print(msg)

            self.save_model(model, name)  # Assuming you have a save_model method

        # Plot the error distribution for the validation set
        if plot_comparison:
            self.plot_comparison()

    def plot_comparison(self) -> None:
        """
        Plots a boxplot comparison of the error distributions for each model evaluated by the `evaluate` method.
        Ensures that only models with recorded errors are included in the plot.
        """

        # Filter names to include only those with recorded errors
        valid_names = [name for name in self.names if name in self.errors]

        # Prepare data for models with recorded errors
        data_to_plot = [self.errors[name] for name in valid_names]

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.boxplot(data_to_plot, labels=valid_names, showmeans=True)

        ax.set_title('Model Error Distribution Comparison')
        ax.set_xlabel('Model')
        ax.set_ylabel('Errors')
        plt.xticks(rotation=45)  # Rotate model names for better readability
        plt.grid(True)
        plt.show()

    def evaluate_validation(self,
                            X_val: pd.DataFrame,  # noqa
                            y_val: pd.Series,
                            verbose: bool = False,
                            plot_comparison: bool = False) -> None:
        """
        Evaluates the models on a separate validation set and optionally plots a comparison of their error distributions.

        Parameters:
            X_val (pd.DataFrame): The input features of the validation set.
            y_val (pd.Series): The target values of the validation set.
            verbose (bool): If True, prints model results; otherwise, remains silent. Defaults to False.
            plot_comparison (bool): If True, plots the boxplot comparison between models; otherwise, does not plot. Defaults to False.
        """

        validation_results = {}
        validation_errors = {}  # To store raw errors for boxplot

        for name, model in self.models:
            predictions = model.predict(X_val)
            errors = predictions - y_val

            model_results = pd.DataFrame({
                'ID': X_val.index,
                'Truth': y_val,
                'Predicted': predictions,
                'Error': errors
            })
            self.detailed_validation_results[name] = model_results

            rmse = np.sqrt(mean_squared_error(y_val, predictions))
            mape = mean_absolute_percentage_error(y_val, predictions) * 100
            validation_errors[name] = errors  # Store raw errors for boxplot

            # Store results
            validation_results[name] = {
                'RMSE': rmse, 'MAPE (%)': mape
            }

            # Print results
            if verbose:
                msg = (f"{name} - Validation RMSE: {rmse:.2f}, "
                       f"Validation MAPE: {mape:.2f}%")
                print(msg)

        # Plot the error distribution for the validation set
        if plot_comparison:
            self.plot_validation_comparison(validation_errors)

    def plot_validation_comparison(self, validation_errors: Dict[str, np.ndarray]) -> None:
        """
        Plots a boxplot comparison of the validation error distributions for each model.

        Parameters:
            validation_errors (Dict[str, np.ndarray]): A dictionary containing the raw errors for each model.
        """
        data_to_plot = [validation_errors[name] for name in self.names]

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.boxplot(data_to_plot, labels=self.names, showmeans=True)

        ax.set_title('Validation Set Model Error Distribution Comparison')
        ax.set_xlabel('Model')
        ax.set_ylabel('Errors')
        plt.xticks(rotation=45)  # Rotate model names for better readability
        plt.grid(True)
        plt.show()

    def plot_learning_curves(self,
                             X: pd.DataFrame,  # noqa
                             y: pd.Series,
                             train_sizes: np.ndarray = np.linspace(0.1, 1.0, 5),
                             scoring: str = 'neg_mean_absolute_error') -> None:
        """
        Plots the learning curves for each model to assess how well the model performs as more data is added.

        Parameters:
            X (pd.DataFrame): The input features for training the model.
            y (pd.Series): The target values for the model training.
            train_sizes (np.ndarray): Proportions of the dataset to generate learning curves for.
            scoring (str): Scoring metric to use for evaluating the model performance.
        """

        for name, model in self.models:
            plt.figure(figsize=(10, 6))

            train_sizes_abs, train_scores, validation_scores = learning_curve(
                estimator=model,
                X=X,
                y=y,
                train_sizes=train_sizes,
                cv=self.cv_folds,
                scoring=scoring,
                n_jobs=-1,
                random_state=self.random_state,
                shuffle=True
            )

            # Calculate mean and standard deviation for training set scores
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)

            # Calculate mean and standard deviation for test set scores
            validation_scores_mean = np.mean(validation_scores, axis=1)
            validation_scores_std = np.std(validation_scores, axis=1)

            # Plot learning curve
            plt.fill_between(train_sizes_abs, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1, color="r")
            plt.fill_between(train_sizes_abs, validation_scores_mean - validation_scores_std,
                             validation_scores_mean + validation_scores_std, alpha=0.1, color="g")
            plt.plot(train_sizes_abs, train_scores_mean, 'o-', color="r", label="Training score")
            plt.plot(train_sizes_abs, validation_scores_mean, 'o-', color="g", label="Cross-validation score")

            plt.title(f'Learning Curve for {name}\nScoring: {scoring}')
            plt.xlabel('Training examples')
            plt.ylabel('Score')
            plt.legend(loc="best")
            plt.grid(True)
            plt.show()

    def save_model(self, model, model_name: str, directory: str = "saved_models") -> None:
        """Saves the model to a specified directory."""
        if not os.path.exists(directory):
            os.makedirs(directory)
        path = os.path.join(directory, model_name + ".joblib")
        joblib.dump(model, path)
        self.model_paths[model_name] = path
        print(f"Model saved to {path}")

    def load_model(self, model_name: str) -> None:
        """Loads a model from the disk."""
        path = self.model_paths.get(model_name, None)
        if path and os.path.exists(path):
            return joblib.load(path)
        else:
            print(f"No saved model found for {model_name}.")
