from name_matching.check_results import ResultsChecker
from name_matching.name_matcher import NameMatcher
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    precision_recall_curve,
)
from sklearn.preprocessing import StandardScaler
from typing import Optional, Protocol
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class sklearnModel(Protocol):
    """
    Protocol for a machine learning model with fit and predict methods.
    """

    def fit(self): ...
    def predict(self): ...


class scaler(Protocol):
    """
    Protocol for a scaler with fit and transform methods.
    """

    def fit(self): ...
    def transform(self): ...


class NameMatchingOptimiser:
    """
    A class for performing name matching and annotating results based on matching scores.
    """

    def __init__(
        self,
        df_matching_data: pd.DataFrame,
        matching_col: str,
        df_to_be_matched: pd.DataFrame,
        to_be_matched_col: str,
        annotated_data: Optional[dict] = None,
        name_matcher: Optional[NameMatcher] = None,
    ) -> None:
        """
        Initializes the NameMatchingOptimiser with required data and matching algorithm.

        Parameters
        ----------
        df_matching_data : pd.DataFrame
            DataFrame containing the matching data.
        matching_col : str
            Column name in `df_matching_data` to match names.
        df_to_be_matched : pd.DataFrame
            DataFrame containing data to be matched.
        to_be_matched_col : str
            Column name in `df_to_be_matched` to match names.
        annotated_data : Optional[dict], default None
            Dictionary of manually annotated data.
        name_matcher : Optional[NameMatcher], default None
            Custom name matching algorithm. If None, a default `NameMatcher` will be used.
        """
        self._df_matching_data = df_matching_data
        self._matching_col = matching_col
        self._df_to_be_matched = df_to_be_matched
        self._to_be_matched_col = to_be_matched_col
        self._annotated_data = annotated_data
        self._features = None
        self._classes = None
        if name_matcher is None:
            self._nm = NameMatcher()
        else:
            self._nm = name_matcher
        self._nm._number_of_matches = self._nm._num_distance_metrics

    def _perform_name_matching(
        self, metrics: Optional[list[str]]
    ) -> pd.Series | pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
        """
        Performs name matching based on the provided distance metrics.

        Parameters
        ----------
        metrics : Optional[list[str]], default None
            List of metrics to be used for name matching.

        Returns
        -------
        pd.Series | pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]
            The result of name matching.

        Raises
        ------
        ValueError
            If no metrics are provided.
        """
        if metrics is None:
            raise ValueError("no metrics supplied!")
        self._nm.load_and_process_master_data(
            self._matching_col, self._df_matching_data
        )
        self._nm.set_distance_metrics(metrics)
        matches = self._nm.match_names(self._df_to_be_matched, self._to_be_matched_col)
        return matches

    def _annotate(self, matches: pd.DataFrame) -> None:
        """
        Annotates the matches with additional information.

        Parameters
        ----------
        matches : pd.DataFrame
            The DataFrame containing the matched results to annotate.
        """
        if self._annotated_data is None:
            self._annotated_data = {}
        filtered_matches = self._preselect_matches(matches)
        rc = ResultsChecker(filtered_matches, annotated_results=self._annotated_data)
        rc.start()
        self._annotated_data.update(rc.annotated_results)

    def _annotate_matches(self, data: pd.DataFrame) -> None:
        """
        Annotates matches based on the best matching scores.

        Parameters:
        ----------
        data : pd.DataFrame
            The DataFrame containing matched results.
        """
        names = data["original_name"]
        score_cols = data.columns.str.contains("score")
        max_col = data.loc[:, score_cols].idxmax(axis=1)
        max_col = max_col.str.split("_")
        max_col = max_col.apply(lambda x: "match_name_" + x[1])
        for key, idx, val in zip(names.values, names.index, max_col.values):
            self._annotated_data[key] = data.loc[idx, val]  # type: ignore

    def _preselect_matches(self, matches: pd.DataFrame) -> pd.DataFrame:
        """
        Filters matches based on score thresholds.

        Parameters
        ----------
        matches : pd.DataFrame
            The DataFrame containing the matched results.

        Returns
        -------
        pd.DataFrame
            The filtered matches with scores between 70% and 100%.
        """

        matches["max_scr"] = matches[
            matches.columns[matches.columns.str.contains("score")]
        ].max(axis=1)

        # select matches between 70 and 100% for manual inspection
        return matches[(matches["max_scr"] > 70) & (matches["max_scr"] < 100)]

    def annotate(self, data_percentage: float = 0.2):
        """
        Annotates a subset of the data.

        Parameters
        ----------
        data_percentage : float, default 0.2
            The percentage of data to be sampled and annotated.
        """
        self._nm._return_algorithms_score = False
        self._nm.load_and_process_master_data(
            self._matching_col, self._df_matching_data
        )
        reduced_data = self._df_to_be_matched.sample(
            frac=data_percentage, replace=False
        )
        annotate_matches = self._nm.match_names(reduced_data, self._to_be_matched_col)
        self._annotate(annotate_matches)  # type: ignore

    def export_annotation(
        self, file_name: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Exports the annotated data to a CSV or returns it as a DataFrame.

        Parameters
        ----------
        file_name : Optional[str], default None
            The file name to save the annotations as a CSV. If None, the annotations are 
            returned as a DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the annotated results.

        Raises
        ------
        ValueError
            If no annotated data has yet been made.
        """
        if not isinstance(self._annotated_data, dict):
            raise ValueError("annotated data has not yet been generated")

        annotations = pd.DataFrame(
            {
                "original_name": self._annotated_data.keys(),
                "match_name": self._annotated_data.values(),
            }
        )
        if file_name is None:
            return annotations
        else:
            annotations.to_csv(file_name, index=False)

    def import_annotation(self, annotations: pd.DataFrame | str) -> None:
        """
        Imports annotations from a CSV file or DataFrame.

        Parameters
        ----------
        annotations : pd.DataFrame | str
            DataFrame or file path containing annotations.
        """
        if isinstance(annotations, str):
            annotations = pd.read_csv(annotations)
        annotations_dict = annotations.set_index("original_name").to_dict()[
            "match_name"
        ]
        if self._annotated_data is None:
            self._annotated_data = annotations_dict
        else:
            self._annotated_data.update(annotations_dict)

    def plot_curves(self) -> None:
        """
        Plots precision-recall curve and the true positive, false positive rate curve
        based on the model's predictions.
        """
        # Get the predicted probabilities for the positive class
        y_scores = self.model.predict_proba(self._X_test)[:, 1]

        # Compute the precision and recall values
        precision, recall, _ = precision_recall_curve(self._y_test, y_scores)

        # Plot the Precision-Recall curve
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color="b", label="Precision-Recall curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.show()

        x = np.linspace(0, 1, 100, True)
        y_true_positives = []
        y_false_positives = []
        for i in x:
            positives = y_scores > i
            y_true_positives.append(
                np.sum(np.bitwise_and(positives, self._y_test == 1))
                / np.sum(self._y_test)
            )
            y_false_positives.append(
                np.sum(np.bitwise_and(positives, self._y_test == 0))
                / (len(self._y_test) - np.sum(self._y_test))
            )

        # Plot the Precision-Recall curve
        plt.figure(figsize=(8, 6))
        plt.plot(x, y_true_positives, color="b", label="True positives")
        plt.plot(x, y_false_positives, color="r", label="False positives")
        plt.xlabel("Threshold")
        plt.ylabel("Percentage")
        plt.legend()
        plt.show()

    def fit(
        self,
        model: sklearnModel = GradientBoostingClassifier,  # type: ignore
        dataScaler: scaler = StandardScaler,  # type: ignore
        model_args: dict | None = None,
        print_results: bool = True,
    ) -> None:
        """
        Fits a model to the name matching data.

        Parameters
        ----------
        model : sklearnModel, default GradientBoostingClassifier
            The model to use for fitting.
        dataScaler : scaler, default StandardScaler
            The scaler to use for feature scaling.
        model_args : Optional[dict], default None
            Arguments for the model used during fitting.
        print_results : bool, default True
            Whether to print evaluation metrics after fitting.
        """
        self._model = model
        # Generate matches
        self._nm._return_algorithms_score = True
        self._nm.load_and_process_master_data(
            self._matching_col, self._df_matching_data
        )
        matches, possible_names = self._nm.match_names(
            self._df_to_be_matched, self._to_be_matched_col
        )

        # Select matches for model training and fitting
        true_index = {}
        false_index = {}
        if isinstance(self._annotated_data, dict):
            for idx, name in enumerate(self._df_to_be_matched[self._to_be_matched_col]):
                if isinstance(self._annotated_data, dict):
                    if name in self._annotated_data.keys():
                        if name != self._annotated_data[name]:
                            temp_idx = possible_names[idx] == self._annotated_data[name]  # type: ignore
                            if np.sum(temp_idx) > 0:
                                true_index[idx] = matches[idx][temp_idx][0]  # type: ignore
                                matches_temp = matches[idx].copy()  # type: ignore
                                matches_temp[temp_idx] = matches_temp[temp_idx] * 0
                                false_index[idx] = matches_temp[
                                    np.argmax(np.mean(matches_temp, axis=1))
                                ] # type: ignore
                            else:
                                print(f"{idx} has no match in names")
        else:
            raise ValueError(
                "Please first annotate some data or provide ",
                "annotated data so the model can be fitted",
            )
        a = np.array(list(true_index.values()))
        b = np.array(list(false_index.values()))
        X = np.vstack((a, b))
        Y = np.hstack([np.ones(len(true_index)), np.zeros(len(false_index))])
        self.scaler = dataScaler().fit(X)  # type: ignore
        features = self.scaler.transform(X)

        if model_args is None:
            self._model_args = {}
        else:
            self._model_args = model_args

        self.model = self._model(**self._model_args)  # type: ignore
        X_train, X_test, y_train, y_test = train_test_split(features, Y, test_size=0.3)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        self._X_test = X_test
        self._y_test = y_test
        self._X_train = X_train
        self._y_train = y_train

        if print_results:
            print(f"accuracy - {accuracy_score(y_test, y_pred)}")
            print(f"recall - {recall_score(y_test, y_pred)}")
            print(f"precision - {precision_score(y_test, y_pred)}")

    def predict(self, threshold: float = 0.5) -> dict:
        """
        Predicts the best matches for the names to be matched.

        Parameters:
        ----------
        threshold : float, default 0.5
            The threshold for considering a match.

        Returns:
        -------
        dict
            A dictionary mapping original names to predicted match names.
        """

        self._nm._return_algorithms_score = True
        self._nm.load_and_process_master_data(
            self._matching_col, self._df_matching_data
        )
        algorithm_scores, names = self._nm.match_names(
            self._df_to_be_matched, self._to_be_matched_col
        )
        features = self.scaler.transform(
            np.stack(algorithm_scores.to_numpy()).reshape(  # type: ignore
                -1, self._nm._num_distance_metrics
            )
        )
        probabilities = self.model.predict_proba(features)[:, 1]
        probabilities = probabilities.reshape(-1, self._nm._top_n)
        results = {}
        for idx, name in enumerate(self._df_to_be_matched[self._to_be_matched_col]):
            best_match = np.argmax(probabilities[idx, :])
            if probabilities[idx, best_match] > threshold:
                results[name] = names[idx, best_match]  # type: ignore
            else:
                results[name] = ""
        return results
