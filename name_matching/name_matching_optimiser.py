from name_matching.check_results import ResultsChecker
from name_matching.name_matcher import NameMatcher
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score
from typing import Protocol
import pandas as pd
import numpy as np


class sklearnModel(Protocol):
    def fit(): ...
    def predict(): ...


class NameMatchingOptimiser:

    def __init__(
        self,
        df_matching_data,
        matching_col,
        df_to_be_matched,
        to_be_matched_col,
        annotated_data: dict | None = None,
    ):
        self._df_matching_data = df_matching_data
        self._matching_col = matching_col
        self._df_to_be_matched = df_to_be_matched
        self._to_be_matched_col = to_be_matched_col
        self._annotated_data = annotated_data
        self._score_cols = []
        self._features = None 
        self._classes = None

    def _perform_name_matching(self, metrics: list[str]|None) -> pd.DataFrame|tuple[pd.DataFrame,pd.DataFrame]:
        if metrics is not None:
            nm = NameMatcher(number_of_matches=len(metrics))
            nm.load_and_process_master_data(self._matching_col, self._df_matching_data)
            nm.set_distance_metrics(metrics)
            matches = nm.match_names(self._df_to_be_matched, self._to_be_matched_col)
            self._score_cols = matches.columns.str.contains('score')
            return matches

    def _annotate(self, matches) -> dict:
        if self._annotated_data is None:
            self._annotated_data = {}
            filtered_matches = self._preselect_matches(matches)
            rc = ResultsChecker(filtered_matches, annotated_results=self._annotated_data)
            rc.start()
        else:
            rc = ResultsChecker(filtered_matches, annotated_results=self._annotated_data)

        return rc.annotated_results

    def _annotate_matches(self, series) -> None:
        self._annotated_data[series.name] = series['match_name_' + series[self._score_cols].idxmax().split('_')[1]]

    def _preselect_matches(self, matches) -> pd.DataFrame:

        matches['max_score'] = matches[matches.columns[matches.columns.str.contains('score')]].sum(axis=1)

        # annotate 100% matches as correct
        matches[matches['max_score']==100].apply(self._annotate_matches, axis=1)

        # select matches between 80 and 100% for manual inspection
        return matches[(matches['max_score']>80) & (matches['max_score']<100)]

    def select_metrics(
        self,
        metrics: list[str] = [
            "indel",
            "discounted_levenshtein",
            "tichy",
            "cormodeL_z",
            "iterative_sub_string",
            "baulieu_xiii",
            "clement",
            "dice_asymmetricI",
            "kuhns_iii",
            "overlap",
            "pearson_ii",
            "weighted_jaccard",
            "warrens_iv",
            "bag",
            "rouge_l",
            "ratcliff_obershelp",
            "ncd_bz2",
            "fuzzy_wuzzy_partial_string",
            "fuzzy_wuzzy_token_sort",
            "fuzzy_wuzzy_token_set",
        ],
        number_of_algorithms: int = 3,
        model: sklearnModel = GradientBoostingRegressor,
    ):
        matches = self._perform_name_matching(None)
        self._annotate(matches)

    def evaluate_model(self, retries: int = 5):
        accuracy = []
        recall = []
        precision = []
        if (self._features is not None) & (self._classes is not None):
            for _ in retries:
                X_train, X_test, y_train, y_test = train_test_split(self._features, self._classes, test_size=0.3)
                mod = self._model(**self._model_args)
                mod.fit(X_train, y_train)
                y_pred = mod.predict(X_test)
                precision.append(precision_score(y_test, y_pred))
                recall.append(recall_score(y_test, y_pred))
                accuracy.append(accuracy_score(y_test, y_pred))

        print(f"accuracy - max: {accuracy.max():.2f} mean: {accuracy.mean():.2f} min: {accuracy.min():.2f}")
        print(f"recall - max: {accuracy.max():.2f} mean: {accuracy.mean():.2f} min: {accuracy.min():.2f}")
        print(f"precision - max: {accuracy.max():.2f} mean: {accuracy.mean():.2f} min: {accuracy.min():.2f}")

        

    def optimise(self, model: sklearnModel = GradientBoostingRegressor, model_args: dict|None=None, check_additional_names:bool=True):
        self._model = model
        nm = NameMatcher(number_of_matches=5, return_algorithms_score=True)
        nm.load_and_process_master_data(self._matching_col, self._df_matching_data)
        matches, possible_names = nm.match_names(self._df_to_be_matched, self._to_be_matched_col)
        if check_additional_names:
            nm_annotate = NameMatcher(number_of_matches=5, return_algorithms_score=False)
            nm_annotate.load_and_process_master_data(self._matching_col, self._df_matching_data)
            annotate_matches = nm_annotate.match_names(self._df_to_be_matched, self._to_be_matched_col)
            annotated = self._annotate(annotate_matches)
        names = np.repeat(nm._df_matching_data[nm._column].to_numpy(), nm._top_n)
        match_names = possible_names.reshape(-1)
        annotated_names = []
        for idx, name in enumerate(names):
            if name in annotated.keys():
                annotated_names.append(annotated[name])
            else:
                annotated_names.append('_________')
        name_filter = annotated_names!='_________'
        classes = (annotated_names == match_names).astype(int)
        self._classes = classes[name_filter]
        features = np.stack(matches.to_numpy()).reshape(-1, nm._num_distance_metrics)
        self._features = features[name_filter, :]

        if model_args is None:
            self._model_args = {}
        else:
            self._model_args = model_args

        mod = self._model(**self._model_args)
        mod.fit(self._features.reshape(-1, nm._num_distance_metrics), self._classes.reshape(-1))

        return mod
