from name_matching.check_results import ResultsChecker
from name_matching.name_matcher import NameMatcher
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, precision_recall_curve
from typing import Protocol
import matplotlib.pyplot as plt
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
        name_matcher: NameMatcher| None = None,
        minimum_mean_algorithm_score: float = 0.8,
    ):
        self._df_matching_data = df_matching_data
        self._matching_col = matching_col
        self._df_to_be_matched = df_to_be_matched
        self._to_be_matched_col = to_be_matched_col
        self._annotated_data = annotated_data
        self._algo_threshold = minimum_mean_algorithm_score
        self._features = None 
        self._classes = None
        if name_matcher is None:
            self._nm = NameMatcher()
        else:
            self._nm = name_matcher
        self._nm._number_of_matches = self._nm._num_distance_metrics

    def _perform_name_matching(self, metrics: list[str]|None) -> pd.Series|pd.DataFrame|tuple[pd.DataFrame,pd.DataFrame]:
        if metrics is None:
            raise ValueError('no metrics supplied!')
        self._nm.load_and_process_master_data(self._matching_col, self._df_matching_data)
        self._nm.set_distance_metrics(metrics)
        matches = self._nm.match_names(self._df_to_be_matched, self._to_be_matched_col)
        return matches

    def _annotate(self, matches):
        if self._annotated_data is None:
            self._annotated_data = {}
        filtered_matches = self._preselect_matches(matches)
        rc = ResultsChecker(filtered_matches, annotated_results=self._annotated_data)
        rc.start()
        self._annotated_data.update(rc.annotated_results)


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
        model: sklearnModel = GradientBoostingClassifier,
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
        self._nm._return_algorithms_score = False
        self._nm.load_and_process_master_data(self._matching_col, self._df_matching_data)
        reduced_data = self._df_to_be_matched.sample(frac=data_percentage, replace=False)
        annotate_matches = self._nm.match_names(reduced_data, self._to_be_matched_col)
        self._annotate(annotate_matches)

    def export_annotation(self, file_name: None|str=None):
        annotations = pd.DataFrame({'original_name':self._annotated_data.keys(),'match_name':self._annotated_data.values()})
        if file_name is None:
            return annotations
        else:
            annotations.to_csv(file_name, index=False)

    def import_annotation(self, annotations: pd.DataFrame|str):
        if isinstance(annotations, str):
            annotations = pd.read_csv(annotations)
        annotations_dict = annotations.set_index('original_name').to_dict()['match_name']
        if self._annotated_data is None:
            self._annotated_data = annotations_dict
        else:
            self._annotated_data.update(annotations_dict)

    def plot_curves(self):
        # Get the predicted probabilities for the positive class
        y_scores = self.model.predict_proba(self._X_test)[:, 1]

        # Compute the precision and recall values
        precision, recall, _ = precision_recall_curve(self._y_test, y_scores)

        # Plot the Precision-Recall curve
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='b', label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.show()
        
        x = np.linspace(0,1,100,True)
        y_true_positives = []
        y_false_positives = []
        for i in x:
            positives = y_scores > i
            y_true_positives.append(np.sum(np.bitwise_and(positives, self._y_test==1))/np.sum(self._y_test))
            y_false_positives.append(np.sum(np.bitwise_and(positives, self._y_test==0))/len(self._y_test))

        # Plot the Precision-Recall curve
        plt.figure(figsize=(8, 6))
        plt.plot(x, y_true_positives, color='b', label='True positives')
        plt.plot(x, y_false_positives, color='r', label='False positives')
        plt.xlabel('Threshold')
        plt.ylabel('Percentage')
        plt.legend()
        plt.show()



    def fit(self, model: sklearnModel = GradientBoostingClassifier, model_args: dict|None=None, print_results: bool=True)->sklearnModel:
        empty_string = '_________'
        self._model = model
        self._nm._return_algorithms_score = True
        self._nm.load_and_process_master_data(self._matching_col, self._df_matching_data)
        matches, possible_names = self._nm.match_names(self._df_to_be_matched, self._to_be_matched_col)
        names = np.repeat(self._df_to_be_matched[self._to_be_matched_col].to_numpy(), self._nm._top_n)
        match_names = possible_names.reshape(-1)
        annotated_names = []
        for idx, name in enumerate(names):
            if name in self._annotated_data.keys():
                annotated_names.append(self._annotated_data[name])
            else:
                annotated_names.append(empty_string)
        name_filter = annotated_names!=empty_string
        classes = (annotated_names == match_names).astype(int)
        self._classes = classes[name_filter]
        features = np.stack(matches.to_numpy()).reshape(-1, self._nm._num_distance_metrics)
        self._features = features[name_filter, :]

        if model_args is None:
            self._model_args = {}
        else:
            self._model_args = model_args

        self.model = self._model(**self._model_args)
        features = self._features.reshape(-1, self._nm._num_distance_metrics)
        classes = self._classes.reshape(-1)[features.mean(axis=1) > self._algo_threshold]
        features = features[features.mean(axis=1) > self._algo_threshold]
        X_train, X_test, y_train, y_test = train_test_split(features, classes, test_size=0.3)
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


    def predict(self, threshold:float = 0.5):

        self._nm._return_algorithms_score = True
        self._nm.load_and_process_master_data(self._matching_col, self._df_matching_data)
        algorithm_scores, names = self._nm.match_names(self._df_to_be_matched, self._to_be_matched_col)
        probabilities = self.model.predict_proba(np.stack(algorithm_scores.to_numpy()).reshape(-1, self._nm._num_distance_metrics))[:,1]
        probabilities = probabilities.reshape(-1, self._nm._top_n)
        results = {}
        for idx, name in enumerate(self._df_to_be_matched[self._to_be_matched_col]):
            best_match = np.argmax(probabilities[idx, :])
            if probabilities[idx, best_match] > threshold:
                results[name] = names[idx, best_match]
            else:
                results[name] = ''
        return results