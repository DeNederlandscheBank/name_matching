"""Simple tool for class labeling of images in Jupyter notebooks.

Classes:
    ImageClassLabeler

"""
import pandas as pd
import ipywidgets as ipyw
from IPython.display import display, clear_output

class ResultsChecker:
    """
    """

    def __init__(
            self,
            results: pd.DataFrame,
            name_col: str = 'original_name',
            annotated_results: dict|None = None,
    ) -> None:
        """
        """
        self._results_data = results
        self._name_col = name_col
        if annotated_results is None:
            self._annotated_results = {}
        else:
            self._annotated_results = annotated_results
            self._check_annotated_results()

    @property
    def annotated_results(self) -> dict:
        return self._annotated_results

    def _check_annotated_results(self):
        pass

    def _initiate_buttons(self):
        button_nm = ipyw.Button(description='no match', layout=ipyw.Layout(width='100%'))
        button_nm.on_click(self._no_match)
        button_dn = ipyw.Button(description='don\'t know', layout=ipyw.Layout(width='100%'))
        button_dn.on_click(self._skip)
        button_stop = ipyw.Button(description='stop', style={'button_color':'white'}, layout=ipyw.Layout(width='100%'))
        button_stop.on_click(self._stop)
        return [button_nm, button_dn, button_stop]

    def _no_match(self, button: ipyw.Button):
        self._annotated_results[self._results_data.loc[self._possible_nodes[self._index], 'original_name']] = -1
        self._skip(None)

    def _possible_names(self):
        row = self._results_data.loc[self._possible_nodes[self._index]]
        return row[row.index.str.contains('match_name')].unique()

    def _skip(self, button: ipyw.Button|None):
        self._index = self._index + 1
        if self._index >= len(self._possible_nodes):
            self._stop(None)
        else:
            buttons = self._initiate_buttons()
            for name in self._possible_names():
                self._add_button(buttons, name)
            self._show_data(buttons)

    def _save_result(self, button):
        self._annotated_results[self._results_data.loc[self._possible_nodes[self._index], 'original_name']] = button.description
        self._skip(None)

    def _add_button(self, buttons: list, name: str):
        button = ipyw.Button(description= name, layout=ipyw.Layout(width='100%'))
        button.on_click(self._save_result)
        buttons.insert(1, button)

    def _show_data(self, buttons: list):
        clear_output()
        label = ipyw.Label(value=f"{self._results_data.loc[self._possible_nodes[self._index],'original_name']}",
                           layout=ipyw.Layout(padding='10px', font_weight='bold', font_size='20px', width='100%', height='50px'))
        buttons.insert(0, label)
        display(ipyw.VBox(children=buttons, layout=ipyw.Layout(width='500px')))

    def _stop(self, button: ipyw.Button):
        clear_output()

    def _define_possible_nodes(self, selection) -> list:

        if selection == 'random':
            return self._results_data.index.to_series().sample(n=len(self._results_data)).to_list()

    def start(self, selection: str = 'random', fresh_start: bool = False):
        self._index = -1
        self._possible_nodes = self._define_possible_nodes(selection)
        if fresh_start:
            self._annotated_results = {}
        self._skip(None)