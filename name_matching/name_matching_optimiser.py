from name_matching.check_results import ResultsChecker
from name_matching.name_matcher import NameMatcher
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from typing import Protocol
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class sklearnModel(Protocol):
    def fit(self): ...
    def predict(self): ...


class scaler(Protocol):
    def fit(self): ...
    def transform(self): ...


class NameMatchingOptimiser:

    def __init__(
        self,
        df_matching_data:pd.DataFrame,
        matching_col:str,
        df_to_be_matched:pd.DataFrame,
        to_be_matched_col:str,
        annotated_data: dict | None = None,
        name_matcher: NameMatcher| None = None,
    ):
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
            self._annotated_data[key] = data.loc[idx, val] # type: ignore

    def _preselect_matches(self, matches) -> pd.DataFrame:

        matches['max_scr'] = matches[matches.columns[matches.columns.str.contains('score')]].max(axis=1)

        # select matches between 70 and 100% for manual inspection
        return matches[(matches['max_scr']>70) & (matches['max_scr']<100)]

    def annotate(self, data_percentage:float=0.2, model_args: dict|None=None):
        self._nm._return_algorithms_score = False
        self._nm.load_and_process_master_data(self._matching_col, self._df_matching_data)
        reduced_data = self._df_to_be_matched.sample(frac=data_percentage, replace=False)
        annotate_matches = self._nm.match_names(reduced_data, self._to_be_matched_col)
        self._annotate(annotate_matches)

    def export_annotation(self, file_name: None|str=None):
        annotations = pd.DataFrame({'original_name':self._annotated_data.keys(),'match_name':self._annotated_data.values()}) # type: ignore
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



    def fit(self, 
            model: sklearnModel = GradientBoostingClassifier,  # type: ignore
            dataScaler: scaler = StandardScaler, # type: ignore
            model_args: dict|None=None, 
            print_results: bool=True)-> None:
        
        self._model = model
        # Generate matches
        self._nm._return_algorithms_score = True
        self._nm.load_and_process_master_data(self._matching_col, self._df_matching_data)
        matches, possible_names = self._nm.match_names(self._df_to_be_matched, self._to_be_matched_col)

        # Select matches for model training and fitting
        true_index = {}
        false_index = {}
        if isinstance(self._annotated_data, dict):
            for idx, name in enumerate(self._df_to_be_matched[self._to_be_matched_col]):
                if isinstance(self._annotated_data, dict):
                    if name in self._annotated_data.keys():
                        if name != self._annotated_data[name]:
                            temp_idx = possible_names[idx] == self._annotated_data[name] # type: ignore
                            if np.sum(temp_idx) > 0:
                                true_index[idx] = matches[idx][temp_idx][0] # type: ignore
                                matches_temp = matches[idx].copy() # type: ignore
                                matches_temp[temp_idx] = matches_temp[temp_idx]*0
                                false_index[idx] = matches_temp[np.argmax(np.mean(matches_temp, axis=1))] # type: ignore
                            else:
                                print(f"{idx} has no match in names")
        else:
            raise ValueError("Please first annotate some data or provide ",
                                "annotated data so the model can be fitted")
        a = np.array(list(true_index.values()))
        b = np.array(list(false_index.values()))
        X = np.vstack((a, b))
        Y = np.hstack([np.ones(len(true_index)), np.zeros(len(false_index))])
        self.scaler = dataScaler().fit(X) # type: ignore
        features = self.scaler.transform(X)

        if model_args is None:
            self._model_args = {}
        else:
            self._model_args = model_args

        self.model = self._model(**self._model_args) # type: ignore
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


    def predict(self, threshold:float = 0.5):

        self._nm._return_algorithms_score = True
        self._nm.load_and_process_master_data(self._matching_col, self._df_matching_data)
        algorithm_scores, names = self._nm.match_names(self._df_to_be_matched, self._to_be_matched_col)
        features = self.scaler.transform(np.stack(algorithm_scores.to_numpy()).reshape(-1, self._nm._num_distance_metrics)) # type: ignore
        probabilities = self.model.predict_proba(features)[:,1]
        probabilities = probabilities.reshape(-1, self._nm._top_n)
        results = {}
        for idx, name in enumerate(self._df_to_be_matched[self._to_be_matched_col]):
            best_match = np.argmax(probabilities[idx, :])
            if probabilities[idx, best_match] > threshold:
                results[name] = names[idx, best_match] # type: ignore
            else:
                results[name] = ''
        return results