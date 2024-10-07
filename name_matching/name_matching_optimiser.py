from name_matching.check_results import ResultsChecker
from name_matching.name_matcher import NameMatcher
from sklearn.ensemble import GradientBoostingRegressor
from typing import Protocol
import pandas as pd


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

    def _perform_name_matching(self, metrics: list[str]|None) -> pd.DataFrame:
        nm = NameMatcher(return_algorithms_score=True)
        if metrics is not None:
            nm.set_distance_metrics(metrics)
        nm.load_and_process_master_data(self._df_matching_data, self._matching_col)
        matches = nm.match_names(self._df_to_be_matched, self._to_be_matched_col)
        return matches

    def _annotate(self, matches):
        if self._annotated_data is None:
            rc = ResultsChecker(matches)
            rc.start()
        else:
            rc = ResultsChecker(matches, annotated_results=self._annotated_data)
        

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

    def optimise(self, model: sklearnModel = GradientBoostingRegressor):
        matches = self._perform_name_matching(None)
        self._annotate(matches)

        model.fit
