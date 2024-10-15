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
        df_matching_data:pd.DataFrame,
        matching_col:str,
        df_to_be_matched:pd.DataFrame,
        to_be_matched_col:str,
        annotated_data: dict | None = None,
    ):
        self._df_matching_data = df_matching_data
        self._matching_col = matching_col
        self._df_to_be_matched = df_to_be_matched
        self._to_be_matched_col = to_be_matched_col
        self._annotated_data = annotated_data
        self._features = None 
        self._classes = None

    def _perform_name_matching(self, metrics: list[str]|None) -> pd.DataFrame|tuple[pd.DataFrame,pd.DataFrame]:
        if metrics is not None:
            nm = NameMatcher(number_of_matches=len(metrics))
            nm.load_and_process_master_data(self._matching_col, self._df_matching_data)
            nm.set_distance_metrics(metrics)
            matches = nm.match_names(self._df_to_be_matched, self._to_be_matched_col)
            return matches

    def _annotate(self, matches):
        if self._annotated_data is None:
            self._annotated_data = {}
            filtered_matches = self._preselect_matches(matches)
            rc = ResultsChecker(filtered_matches)
            rc.start()
            self._annotated_data.update(rc.annotated_results)
        else:
            rc = ResultsChecker(filtered_matches, annotated_results=self._annotated_data)


    def _annotate_matches(self, data: pd.DataFrame) -> None:
        names = data['original_name']
        score_cols = data.columns.str.contains('score')
        max_col = data.loc[:, score_cols].idxmax(axis=1)
        max_col = max_col.str.split('_')
        max_col = max_col.apply(lambda x: 'match_name_' + x[1])
        for key, idx, val in zip(names.values, names.index, max_col.values):
            self._annotated_data[key] = data.loc[idx, val]

    def _preselect_matches(self, matches) -> pd.DataFrame:

        matches['max_scr'] = matches[matches.columns[matches.columns.str.contains('score')]].max(axis=1)

        # annotate 100% matches as correct
        self._annotate_matches(matches[matches['max_scr']==100])

        # select matches between 80 and 100% for manual inspection
        return matches[(matches['max_scr']>70) & (matches['max_scr']<100)]

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


    def annotate(self, data_percentage:float=0.2, model_args: dict|None=None):
        nm_annotate = NameMatcher(number_of_matches=5, return_algorithms_score=False)
        nm_annotate.load_and_process_master_data(self._matching_col, self._df_matching_data)
        reduced_data = self._df_to_be_matched.sample(frac=data_percentage, replace=False)
        annotate_matches = nm_annotate.match_names(reduced_data, self._to_be_matched_col)
        self._annotate(annotate_matches)


    def optimise(self, model: sklearnModel = GradientBoostingRegressor, model_args: dict|None=None)->sklearnModel:
        self._model = model
        nm = NameMatcher(number_of_matches=5, return_algorithms_score=True)
        nm.load_and_process_master_data(self._matching_col, self._df_matching_data)
        matches, possible_names = nm.match_names(self._df_to_be_matched, self._to_be_matched_col)
        names = np.repeat(nm._df_matching_data[nm._column].to_numpy(), nm._top_n)
        match_names = possible_names.reshape(-1)
        annotated_names = []
        for idx, name in enumerate(names):
            if name in self._annotated_data.keys():
                annotated_names.append(self._annotated_data[name])
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
