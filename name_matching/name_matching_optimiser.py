import copy
import random
from name_matching.check_results import ResultsChecker
from name_matching.name_matcher import NameMatcher
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from typing import Any, Optional, Protocol
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
        self._true_index = None
        self._false_index = None
        self._all_false = None

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
            self._matching_col, self._df_matching_data, False
        )
        self._nm.set_distance_metrics(metrics)
        matches = self._nm.match_names(self._df_to_be_matched, self._to_be_matched_col)
        return matches

    def _annotate(self, matches: pd.DataFrame, lower_bound: float) -> None:
        """
        Annotates the matches with additional information.

        Parameters
        ----------
        matches : pd.DataFrame
            The DataFrame containing the matched results to annotate.
        lower_bound : float
            The lower bound for the matches to be served for annotation
        """
        if self._annotated_data is None:
            self._annotated_data = {}
        filtered_matches = self._preselect_matches(matches, lower_bound)
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

    @staticmethod
    def split_list_random_uneven(lst, x):
        """
        Split a list into `x` randomly shuffled, nearly equal-length sublists.

        This function shuffles the input list randomly and then divides it into
        `x` sublists, where the sublists are as evenly sized as possible. 
        The remainder (if any) is distributed such that the first `m` sublists
        get one extra element.

        Parameters
        ----------
        lst : list
            The list to be split into sublists.
        x : int
            The number of sublists to create.

        Returns
        -------
        list of lists
            A list containing `x` sublists with elements randomly selected 
            from the original list.
        """
        lst = copy.deepcopy(lst)
        random.shuffle(lst)
        k, m = divmod(len(lst), x)
        return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(x)]

    def cross_validate_model(self, model_name: Optional[str] = None, threshold:float=0.5, n_folds=5) -> pd.DataFrame:
        accuracy = []
        recall = []
        precision = []
        f1 = []

        if self._true_index is None:
            raise ValueError("Please fit the model first before running the cross validation")
        lists = self.split_list_random_uneven(list(self._true_index.keys()), n_folds)

        for train_idx in lists:                
            X_train, y_train, X_test, y_test = self._generate_balanced_split(0.8, train_idx)
            self.model.fit(X_train, y_train)

            y_pred = self.model.predict_proba(X_test)[:, 1]
            y_pred[y_pred >= threshold] = 1
            y_pred[y_pred < threshold] = 0
            accuracy.append(accuracy_score(y_test.reshape(-1), y_pred))
            recall.append(recall_score(y_test.reshape(-1), y_pred))
            precision.append(precision_score(y_test.reshape(-1), y_pred))
            f1.append(f1_score(y_test.reshape(-1), y_pred))            

        if model_name is None:
            model_name = str(self.model).split("(")[0]
        ind = pd.MultiIndex.from_tuples(
            [
                ("Accuracy", "\mu"),
                ("Accuracy", "$\sigma$"),
                ("F1", "\mu"),
                ("F1", "$\sigma$"),
                ("Precision", "\mu"),
                ("Precision", "$\sigma$"),
                ("Recall", "\mu"),
                ("Recall", "$\sigma$"),
            ]
        )
        df = pd.DataFrame(index=ind, columns=[model_name])

        df.loc[("Accuracy", "\mu"), model_name] = np.mean(accuracy) # type: ignore
        df.loc[("Accuracy", "$\sigma$"), model_name] = np.std(accuracy) # type: ignore
        df.loc[("F1", "\mu"), model_name] = np.mean(f1) # type: ignore
        df.loc[("F1", "$\sigma$"), model_name] = np.std(f1) # type: ignore
        df.loc[("Precision", "\mu"), model_name] = np.mean(precision) # type: ignore
        df.loc[("Precision", "$\sigma$"), model_name] = np.std(precision) # type: ignore
        df.loc[("Recall", "\mu"), model_name] = np.mean(recall) # type: ignore
        df.loc[("Recall", "$\sigma$"), model_name] = np.std(recall) # type: ignore

        return df

    def _preselect_matches(
        self, matches: pd.DataFrame, lower_bound: float
    ) -> pd.DataFrame:
        """
        Filters matches based on score thresholds.

        Parameters
        ----------
        matches : pd.DataFrame
            The DataFrame containing the matched results.
        lower_bound : float
            The lower bound for the matches to be served for annotation

        Returns
        -------
        pd.DataFrame
            The filtered matches with scores between lower_bound and 100%.
        """

        matches["max_scr"] = matches[
            matches.columns[matches.columns.str.contains("score")]
        ].max(axis=1)

        # select matches between lower_bound and 100% for manual inspection
        return matches[(matches["max_scr"] > lower_bound) & (matches["max_scr"] < 100)]

    def annotate(
        self,
        lower_bound: float = 70.0,
        data_percentage: float = 0.2,
        max_matches: int = 10,
    ):
        """
        Annotates a subset of the data.

        Parameters
        ----------
        data_percentage : float, default 0.2
            The percentage of data to be sampled and annotated.
        lower_bound : float
            The lower bound for the matches to be served for annotation
            default = 70.0
        """
        self._nm._return_algorithms_score = False
        if max_matches > self._nm._top_n:
            max_matches = self._nm._top_n
        self._nm._number_of_matches = max_matches
        self._nm.load_and_process_master_data(
            self._matching_col, self._df_matching_data, False
        )
        reduced_data = self._df_to_be_matched.sample(
            frac=data_percentage, replace=False
        )
        annotate_matches = self._nm.match_names(reduced_data, self._to_be_matched_col)
        self._annotate(annotate_matches, lower_bound)  # type: ignore

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

    def plot_curves(self, absolute: bool = True, return_data: bool = False) -> None|dict[str, Any]:
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

        x = np.linspace(0, 1, 1000, True)
        y_true_positives = []
        y_false_positives = []
        for i in x:
            positives = y_scores > i
            y_true_positives.append(
                np.sum(np.bitwise_and(positives, self._y_test == 1))
            )
            y_false_positives.append(
                np.sum(np.bitwise_and(positives, self._y_test == 0))
            )
            if not absolute:
                y_true_positives[-1] = y_true_positives[-1] / np.sum(self._y_test)
                y_false_positives[-1] = y_false_positives[-1] / (
                    len(self._y_test) - np.sum(self._y_test)
                )

        # Plot the true-false positives curve
        plt.figure(figsize=(8, 6))
        plt.plot(x, y_true_positives, color="b", label="True positives")
        plt.plot(x, y_false_positives, color="r", label="False positives")
        plt.xlabel("Threshold")
        if absolute:
            plt.ylabel("Number")
        else:
            plt.ylabel("Percentage")
        plt.ylim((0, np.sum(self._y_test) + 1))
        plt.legend()
        plt.show()

        if return_data:
            return {"true_positives": y_true_positives,
                    "false_positives": y_false_positives,
                    "threshold": x,
                    "Precision": precision,
                    "Recall": recall,
                    }
        
    def _preprocess_fit_annotations(self, dataScaler):

        self._all_false = {}
        self._true_index = {}
        self._false_index = {}

        # Generate matches
        self._nm._return_algorithms_score = True
        self._nm.load_and_process_master_data(
            self._matching_col, self._df_matching_data, False
        )
        matches, possible_names = self._nm.match_names(
            self._df_to_be_matched, self._to_be_matched_col
        )
        matches = matches.reset_index(drop=True) # type: ignore
        self.scaler = dataScaler().fit(matches[0])  # type: ignore
        
        self._true_index = {}
        self._false_index = {}
        self._all_false = {}
        idx_true = 0
        idx_false = 0
        if isinstance(self._annotated_data, dict):
            for idx, name in enumerate(self._df_to_be_matched[self._to_be_matched_col]):
                if name in self._annotated_data.keys():
                    if name != self._annotated_data[name]:
                        temp_idx = possible_names[idx] == self._annotated_data[name]  # type: ignore
                        if np.sum(temp_idx) > 0:
                            self._true_index[idx_true] = matches[idx][temp_idx][0]  # type: ignore
                            self._all_false[idx_true] = np.delete(matches[idx].copy(), temp_idx, 0)  # type: ignore
                            self._false_index[idx_true] = self._all_false[idx_true][
                                np.argmax(np.mean(self._all_false[idx_true], axis=1))
                            ]  # type: ignore
                            idx_true = idx_true + 1
                        elif self._annotated_data[name] == -1:
                            self._false_index[idx_false] = matches[idx][np.argmax(np.mean(matches[idx], axis=1))]  # type: ignore
                            idx_false = idx_false + 1
        else:
            raise ValueError(
                "Please first annotate some data or provide ",
                "annotated data so the model can be fitted",
            )


    def _generate_balanced_split(self, train_split:float, train_idx: Optional[list]=None):
        if self._true_index is None or self._false_index is None or self._all_false is None:
            raise ValueError()
        
        if train_idx is None:
            train_idx = np.random.choice(
                list(self._true_index.keys()), int(train_split * len(self._true_index))
            ) # type: ignore
        test_idx = list(set(self._true_index.keys()) - set(train_idx)) # type: ignore
        
        if train_idx is not None:

            X_train = np.array(self._true_index[train_idx[0]])
            X_train = np.vstack([X_train, np.array(self._false_index[train_idx[0]])])
            y_train = np.ones(1)
            y_train = np.vstack([y_train, np.zeros(1)])
            for idx in train_idx[1:]:
                X_train = np.vstack([X_train, np.array(self._true_index[idx])])
                X_train = np.vstack([X_train, np.array(self._false_index[idx])])
                y_train = np.vstack([y_train, np.ones(1)])
                y_train = np.vstack([y_train, np.zeros(1)])

            X_test = np.array(self._true_index[test_idx[0]])
            y_test = np.ones(1)
            X_test = np.vstack([X_test, self._all_false[test_idx[0]]])
            y_test = np.vstack(
                [y_test, np.zeros([len(self._all_false[test_idx[0]]), 1])]
            )
            for idx in test_idx[1:]:
                X_test = np.vstack([X_test, np.array(self._true_index[idx])])
                y_test = np.vstack([y_test, np.ones(1)])
                X_test = np.vstack([X_test, self._all_false[idx]])
                y_test = np.vstack([y_test, np.zeros([len(self._all_false[idx]), 1])])

            X_train = self.scaler.transform(X_train)
            X_test = self.scaler.transform(X_test)
            y_train = y_train.reshape(-1)
            y_test = y_test.reshape(-1)

            return X_train, y_train, X_test, y_test
        raise ValueError()


    def fit(
        self,
        model: sklearnModel = GradientBoostingClassifier,  # type: ignore
        dataScaler: scaler = StandardScaler,  # type: ignore
        model_args: dict | None = None,
        print_results: bool = True,
        train_split: float = 0.7,
        preprocess: bool = False,
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
        train_split : float, default 0.7
            The percentage of the data to be used for training
        """

        self._model = model
        if preprocess or self._true_index is None:
            self._preprocess_fit_annotations(dataScaler)
        self._X_train, self._y_train, self._X_test, self._y_test = self._generate_balanced_split(train_split)

        if model_args is None:
            self._model_args = {}
        else:
            self._model_args = model_args

        self.model = self._model(**self._model_args)  # type: ignore
        self.model.fit(self._X_train, self._y_train)
        y_pred = self.model.predict(self._X_test)

        if print_results:
            print(f"accuracy - {accuracy_score(self._y_test, y_pred)}")
            print(f"recall - {recall_score(self._y_test, y_pred)}")
            print(f"precision - {precision_score(self._y_test, y_pred)}")

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
            self._matching_col, self._df_matching_data, False
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
