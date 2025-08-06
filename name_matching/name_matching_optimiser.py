import copy
import random
from name_matching.match_annotator import MatchAnnotator
from name_matching.name_matcher import NameMatcher
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    precision_recall_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from typing import Any, Optional, Protocol
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class sklearnModel(Protocol):
    """Protocol for a scikit-learn–style estimator.

    A model must implement the `fit` and `predict` methods.
    """

    def fit(self, X: Any, y: Any) -> Any: ...
    def predict(self, X: Any) -> Any: ...


class scaler(Protocol):
    """Protocol for a scaler.

    A scaler must implement the `fit` and `transform` methods.
    """

    def fit(self, X: Any, y: Any = None) -> Any: ...
    def transform(self, X: Any) -> Any: ...


class NameMatchingOptimiser:
    """Optimises name matching using ML annotation and classification.

    Allows annotating matches, training a classifier on annotated data,
    cross-validation, plotting performance curves, and predicting new matches.
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
        """Initialise with master and target DataFrames and optional annotations.

        Parameters
        ----------
        df_matching_data : pd.DataFrame
            DataFrame containing the master list of names to match against.
        matching_col : str
            Column in `df_matching_data` containing the names.
        df_to_be_matched : pd.DataFrame
            DataFrame containing the names requiring matches.
        to_be_matched_col : str
            Column in `df_to_be_matched` containing those names.
        annotated_data : dict, optional
            Pre-annotated mapping of original names to matched names.
        name_matcher : NameMatcher, optional
            Custom NameMatcher instance. If None, a default is created.

        Attributes
        ----------
        _nm : NameMatcher
            Internal name matcher instance.
        _annotated_data : dict
            Stores annotated results.
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
    ) -> pd.DataFrame | pd.Series | tuple[pd.DataFrame, pd.DataFrame]:
        """Perform name matching with specified distance metrics.

        Parameters
        ----------
        metrics : list of str
            List of distance metrics to use (e.g. ['levenshtein', 'jaro']).

        Returns
        -------
        DataFrame or Series or tuple of DataFrames
            If multiple distances are returned separately, returns a tuple;
            otherwise returns a DataFrame or Series of match scores.

        Raises
        ------
        ValueError
            If `metrics` is None or empty.
        """
        if metrics is None:
            raise ValueError("No metrics supplied!")
        self._nm.load_and_process_master_data(
            self._matching_col, self._df_matching_data, False
        )
        self._nm.set_distance_metrics(metrics)
        matches = self._nm.match_names(self._df_to_be_matched, self._to_be_matched_col)
        return matches

    def _annotate(self, matches: pd.DataFrame, lower_bound: float) -> None:
        """Launch manual annotation on a slice of matches above a threshold.

        Parameters
        ----------
        matches : pd.DataFrame
            DataFrame containing candidate matches with score columns.
        lower_bound : float
            Minimum score (%) for considering a pair for annotation.
        """
        if self._annotated_data is None:
            self._annotated_data = {}
        filtered = self._preselect_matches(matches, lower_bound)
        rc = MatchAnnotator(filtered, annotated_results=self._annotated_data)
        rc.start()
        self._annotated_data.update(rc.annotated_results)

    def _annotate_matches(self, data: pd.DataFrame) -> None:
        """Automatically annotate each original name with its highest-score match.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame of matches with one column for each score and
            columns 'original_name' and 'match_name_X'.
        """
        names = data["original_name"]
        score_cols = data.columns.str.contains("score")
        max_col = data.loc[:, score_cols].idxmax(axis=1)
        max_col = max_col.str.split("_").apply(lambda parts: "match_name_" + parts[1])
        for orig, idx, best in zip(names.values, names.index, max_col.values):
            self._annotated_data[orig] = data.loc[idx, best]  # type: ignore

    @staticmethod
    def split_list_random_uneven(lst: list[Any], x: int) -> list[list[Any]]:
        """Shuffle `lst` and split into `x` roughly equal parts.

        Elements are randomly distributed; remainders are distributed one by one
        to the first few sublists.

        Parameters
        ----------
        lst : list
            Input list to split.
        x : int
            Number of sublists to return.

        Returns
        -------
        list of lists
            A list containing `x` sublists of `lst`.
        """
        lst_copy = copy.deepcopy(lst)
        random.shuffle(lst_copy)
        k, m = divmod(len(lst_copy), x)
        return [
            lst_copy[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(x)
        ]

    def cross_validate_model(
        self, model_name: Optional[str] = None, threshold: float = 0.5, n_folds: int = 5
    ) -> pd.DataFrame:
        """Perform `n_folds` cross-validation on the fitted model.

        Parameters
        ----------
        model_name : str, optional
            Name to assign to the model in the results table.
        threshold : float, default=0.5
            Probability threshold for binary decisions.
        n_folds : int, default=5
            Number of folds for random splitting.

        Returns
        -------
        pd.DataFrame
            MultiIndex DataFrame containing mean and std for accuracy,
            precision, recall, and F1 across folds.

        Raises
        ------
        ValueError
            If the model has not been fitted (no `self._true_index`).
        """
        if self._true_index is None:
            raise ValueError("Please fit the model before running cross-validation.")

        accuracy, recall, precision, f1 = [], [], [], []
        lists = self.split_list_random_uneven(list(self._true_index.keys()), n_folds)

        for train_idx in lists:
            X_train, y_train, X_test, y_test = self._generate_balanced_split(
                0.8, train_idx
            )
            model = clone(self.model)
            model.fit(X_train, y_train)

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_test)[:, 1]
                y_pred = (probs >= threshold).astype(int)
            else:
                y_pred = model.predict(X_test)

            accuracy.append(accuracy_score(y_test, y_pred))
            recall.append(recall_score(y_test, y_pred))
            precision.append(precision_score(y_test, y_pred))
            f1.append(f1_score(y_test, y_pred))

        if model_name is None:
            model_name = type(self.model).__name__
        index = pd.MultiIndex.from_tuples(
            [
                ("Accuracy", "μ"),
                ("Accuracy", "σ"),
                ("F1", "μ"),
                ("F1", "σ"),
                ("Precision", "μ"),
                ("Precision", "σ"),
                ("Recall", "μ"),
                ("Recall", "σ"),
            ]
        )
        df = pd.DataFrame(index=index, columns=[model_name])
        df.loc[("Accuracy", "μ"), model_name] = np.mean(accuracy)
        df.loc[("Accuracy", "σ"), model_name] = np.std(accuracy)
        df.loc[("F1", "μ"), model_name] = np.mean(f1)
        df.loc[("F1", "σ"), model_name] = np.std(f1)
        df.loc[("Precision", "μ"), model_name] = np.mean(precision)
        df.loc[("Precision", "σ"), model_name] = np.std(precision)
        df.loc[("Recall", "μ"), model_name] = np.mean(recall)
        df.loc[("Recall", "σ"), model_name] = np.std(recall)

        return df

    def _preselect_matches(
        self, matches: pd.DataFrame, lower_bound: float
    ) -> pd.DataFrame:
        """Filter matches to those with max score between `lower_bound` and 100.

        Parameters
        ----------
        matches : pd.DataFrame
            All candidate matches with one or more ‘score’ columns.
        lower_bound : float
            Minimum acceptable score (exclusive).

        Returns
        -------
        pd.DataFrame
            Subset of `matches` with `lower_bound < max(score_cols) < 100`.
        """
        score_mask = matches.columns.str.contains("score")
        matches["max_scr"] = matches.loc[:, score_mask].max(axis=1)
        return matches[(matches["max_scr"] > lower_bound) & (matches["max_scr"] < 100)]

    def annotate(
        self,
        lower_bound: float = 70.0,
        data_percentage: float = 0.2,
        max_matches: int = 10,
    ) -> None:
        """Sample a subset of the target names and manually annotate matches.

        Parameters
        ----------
        lower_bound : float, default=70.0
            Minimum match score (%) for prompting annotation.
        data_percentage : float, default=0.2
            Fraction of `df_to_be_matched` to sample for annotation.
        max_matches : int, default=10
            Maximum number of candidate matches per name to present.
        """
        self._nm._return_algorithms_score = False
        max_matches = min(max_matches, self._nm._top_n)
        self._nm._number_of_matches = max_matches
        self._nm.load_and_process_master_data(
            self._matching_col, self._df_matching_data, False
        )
        reduced = self._df_to_be_matched.sample(frac=data_percentage, replace=False)
        candidates = self._nm.match_names(reduced, self._to_be_matched_col)
        self._annotate(candidates, lower_bound)  # type: ignore

    def export_annotation(
        self, file_name: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """Export the current annotations to CSV or return as DataFrame.

        Parameters
        ----------
        file_name : str, optional
            Path to save the CSV. If None, returns a DataFrame instead.

        Returns
        -------
        pd.DataFrame or None
            If `file_name` is None, returns a DataFrame of annotations.

        Raises
        ------
        ValueError
            If no annotations exist.
        """
        if not isinstance(self._annotated_data, dict):
            raise ValueError("No annotated data available.")

        df = pd.DataFrame(
            {
                "original_name": list(self._annotated_data.keys()),
                "match_name": list(self._annotated_data.values()),
            }
        )
        if file_name:
            df.to_csv(file_name, index=False)
            return None
        return df

    def import_annotation(self, annotations: pd.DataFrame | str) -> None:
        """Load annotations from a DataFrame or a CSV file.

        Parameters
        ----------
        annotations : pd.DataFrame or str
            If DataFrame, must have columns `original_name` and `match_name`.
            If str, interpreted as CSV file path.
        """
        if isinstance(annotations, str):
            annotations = pd.read_csv(annotations)
        ann_dict = annotations.set_index("original_name")["match_name"].to_dict()
        if self._annotated_data is None:
            self._annotated_data = ann_dict
        else:
            self._annotated_data.update(ann_dict)

    def plot_curves(
        self, absolute: bool = True, return_data: bool = False
    ) -> None | dict[str, Any]:
        """Plot Precision-Recall curve and threshold-based TP/FP/ TN/FN curves.

        Parameters
        ----------
        absolute : bool, default=True
            If True, plot absolute counts; if False, plot normalized rates.
        return_data : bool, default=False
            If True, return the underlying curve data.

        Returns
        -------
        dict or None
            If `return_data` is True, returns a dict with arrays for
            ‘true_positives’, ‘false_positives’, ‘true_negatives’,
            ‘false_negatives’, ‘threshold’, ‘Precision’, and ‘Recall’.
        """
        y_scores = self.model.predict_proba(self._X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(self._y_test, y_scores)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label="Precision-Recall curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.show()

        thresholds = np.linspace(0, 1, 1000)
        tp, fp, tn, fn = [], [], [], []
        for t in thresholds:
            pos = y_scores >= t
            neg = ~pos
            tp.append(np.sum(pos & (self._y_test == 1)))
            fp.append(np.sum(pos & (self._y_test == 0)))
            tn.append(np.sum(neg & (self._y_test == 0)))
            fn.append(np.sum(neg & (self._y_test == 1)))
            if not absolute:
                total_pos = np.sum(self._y_test == 1)
                total_neg = np.sum(self._y_test == 0)
                tp[-1] /= total_pos
                fp[-1] /= total_neg
                tn[-1] /= total_neg
                fn[-1] /= total_pos

        plt.figure(figsize=(8, 6))
        plt.plot(thresholds, tp, label="True positives")
        plt.plot(thresholds, fp, label="False positives")
        plt.xlabel("Threshold")
        plt.ylabel("Count" if absolute else "Rate")
        plt.ylim(0, (np.sum(self._y_test == 1) if absolute else 1))
        plt.legend()
        plt.show()

        if return_data:
            return {
                "true_positives": tp,
                "false_positives": fp,
                "true_negatives": tn,
                "false_negatives": fn,
                "threshold": thresholds,
                "Precision": precision,
                "Recall": recall,
            }

    def _preprocess_fit_annotations(self, dataScaler: type[scaler]) -> None:
        """Prepare training and test indices from annotated matches.

        Runs name matching on all data, then splits each annotated example
        into one positive and one or more negative feature vectors.

        Parameters
        ----------
        dataScaler : class
            Scaler class (e.g. StandardScaler) to fit on all feature vectors.

        Raises
        ------
        ValueError
            If no annotated data is available.
        """
        self._true_index = {}
        self._false_index = {}
        self._all_false = {}

        if not isinstance(self._annotated_data, dict):
            raise ValueError("Annotated data required before fitting.")

        # Generate all candidate scores and names
        self._nm._return_algorithms_score = True
        self._nm.load_and_process_master_data(
            self._matching_col, self._df_matching_data, False
        )
        matches, possible = self._nm.match_names(
            self._df_to_be_matched, self._to_be_matched_col
        )
        matches = matches.reset_index(drop=True)  # type: ignore

        # Fit scaler on full set
        self.scaler = dataScaler().fit(matches[0])  # type: ignore

        idx_true = idx_false = 0
        for idx, name in enumerate(self._df_to_be_matched[self._to_be_matched_col]):
            if name in self._annotated_data:
                true_match = self._annotated_data[name]
                pos_mask = possible[idx] == true_match  # type: ignore
                if np.any(pos_mask):
                    self._true_index[idx_true] = matches[idx][pos_mask][0]  # type: ignore
                    all_other = np.delete(matches[idx], pos_mask, axis=0)  # type: ignore
                    self._all_false[idx_true] = all_other
                    self._false_index[idx_true] = all_other[np.argmax(all_other.mean(axis=1))]  # type: ignore
                    idx_true += 1
                elif true_match == -1:
                    best = matches[idx][np.argmax(matches[idx].mean(axis=1))]  # type: ignore
                    self._false_index[idx_false] = best
                    idx_false += 1

    def _generate_balanced_split(
        self, train_split: float, train_idx: Optional[list[int]] = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build balanced train/test sets from true and false example indices.

        For each annotated index, pairs a true vector with one false vector.

        Parameters
        ----------
        train_split : float
            Fraction of `self._true_index` keys to include in training.
        train_idx : list of int, optional
            Specific keys to use for training; if None, chosen randomly.

        Returns
        -------
        X_train : ndarray
        y_train : ndarray
        X_test : ndarray
        y_test : ndarray

        Raises
        ------
        ValueError
            If `self._true_index`, `self._false_index`, or `self._all_false` is None.
        """
        if (
            self._true_index is None
            or self._false_index is None
            or self._all_false is None
        ):
            raise ValueError("Fit preprocessing must be run first.")

        keys = list(self._true_index.keys())
        if train_idx is None:
            n_train = int(train_split * len(keys))
            train_idx = list(np.random.choice(keys, n_train, replace=False))
        test_idx = [k for k in keys if k not in train_idx]

        # Build training arrays
        X_train = np.vstack(
            [self._true_index[i] for i in train_idx]
            + [self._false_index[i] for i in train_idx]
        )
        y_train = np.hstack([np.ones(len(train_idx)), np.zeros(len(train_idx))])

        # Build test arrays
        X_test = np.vstack(
            [self._true_index[i] for i in test_idx]
            + [self._all_false[i] for i in test_idx]
        )
        y_test = np.hstack(
            [
                np.ones(len(test_idx)),
                np.zeros(sum(len(self._all_false[i]) for i in test_idx)),
            ]
        )

        X_train = self.scaler.transform(X_train)
        X_test = self.scaler.transform(X_test)

        return X_train, y_train, X_test, y_test

    def fit(
        self,
        model: sklearnModel = GradientBoostingClassifier,
        dataScaler: scaler = StandardScaler,
        model_args: Optional[dict] = None,
        print_results: bool = True,
        train_split: float = 0.7,
        preprocess: bool = False,
    ) -> None:
        """Fit a classifier on annotated name-matching examples.

        This will preprocess annotations (if needed), split into train/test,
        scale features, and fit the chosen model.

        Parameters
        ----------
        model : estimator class, default=GradientBoostingClassifier
            Scikit-learn compatible classifier to fit.
        dataScaler : scaler class, default=StandardScaler
            Scaler to normalise feature vectors.
        model_args : dict, optional
            Keyword arguments to pass to the model constructor.
        print_results : bool, default=True
            If True, print accuracy, recall, and precision on the test set.
        train_split : float, default=0.7
            Fraction of data to allocate to training vs. testing.
        preprocess : bool, default=False
            If True or if no prior preprocessing, rerun annotation preprocessing.
        """
        self._model = model
        if preprocess or self._true_index is None:
            self._preprocess_fit_annotations(dataScaler)

        self._X_train, self._y_train, self._X_test, self._y_test = (
            self._generate_balanced_split(train_split)
        )

        self._model_args = model_args or {}
        self.model = self._model(**self._model_args)  # type: ignore
        self.model.fit(self._X_train, self._y_train)

        if print_results:
            y_pred = self.model.predict(self._X_test)
            print(f"accuracy - {accuracy_score(self._y_test, y_pred)}")
            print(f"recall   - {recall_score(self._y_test, y_pred)}")
            print(f"precision- {precision_score(self._y_test, y_pred)}")

    def predict(self, threshold: float = 0.5) -> dict[str, str]:
        """Match new names by applying the trained classifier to all candidates.

        Parameters
        ----------
        threshold : float, default=0.5
            Probability cutoff above which a candidate is accepted.

        Returns
        -------
        dict
            Mapping from each original name in `df_to_be_matched` to the
            best match name (or empty string if below threshold).
        """
        self._nm._return_algorithms_score = True
        self._nm.load_and_process_master_data(
            self._matching_col, self._df_matching_data, False
        )
        scores, names = self._nm.match_names(
            self._df_to_be_matched, self._to_be_matched_col
        )
        feats = np.stack(scores.to_numpy()).reshape(-1, self._nm._num_distance_metrics)
        feats = self.scaler.transform(feats)
        probs = self.model.predict_proba(feats)[:, 1]
        probs = probs.reshape(-1, self._nm._top_n)

        results: dict[str, str] = {}
        for i, orig in enumerate(self._df_to_be_matched[self._to_be_matched_col]):
            best = np.argmax(probs[i])
            results[orig] = names[i, best] if probs[i, best] > threshold else ""
        return results
