import numpy as np
import pandas as pd
import ipywidgets as ipyw
from IPython.display import display, clear_output


class MatchAnnotator:
    """A class to annotate matches. The annotated matches can be used to automate the
    decision on whether something is a correct match or not."""

    def __init__(
        self,
        results: pd.DataFrame,
        name_col: str = "original_name",
        annotated_results: dict | None = None,
    ) -> None:
        """Initialize the MatchAnnotator with results and settings.

        Parameters
        ----------
        results : type
            Description of results.
        name_col : type
            Description of name_col.
        annotated_results : type
            Description of annotated_results.
        """
        self._results_data = results
        self._name_col = name_col
        if annotated_results is None:
            self._annotated_results = {}
        else:
            self._annotated_results = annotated_results

    @property
    def annotated_results(self) -> dict:
        """Return the dictionary of annotated results.


        Returns
        -------
        dict
            Description of return value.
        """
        return self._annotated_results

    def _initiate_buttons(self) -> list:
        """Initialize and return annotation control buttons.


        Returns
        -------
        list
            Description of return value.
        """
        button_nm = ipyw.Button(
            description="no match", layout=ipyw.Layout(width="100%")
        )
        button_nm.on_click(self._no_match)
        button_dn = ipyw.Button(
            description="don't know", layout=ipyw.Layout(width="100%")
        )
        button_dn.on_click(self._skip)
        button_stop = ipyw.Button(
            description="stop",
            style={"button_color": "white"},
            layout=ipyw.Layout(width="100%"),
        )
        button_stop.on_click(self._stop)

        return [button_nm, button_dn, button_stop]

    def _no_match(self, button: ipyw.Button) -> None:
        """Handle 'no match' button click and skip to next item.

        Parameters
        ----------
        button : ipyw.Button
            Description of button.
        """
        self._annotated_results[
            self._results_data.loc[self._possible_nodes[self._index], "original_name"]
        ] = -1
        self._skip(None)

    def _possible_names(self) -> np.ndarray:
        """Retrieve possible match names for the current item.

        Returns
        -------
        np.ndarray : type
            Description of return value.
        """
        row = self._results_data.loc[self._possible_nodes[self._index]]
        return row[row.index.str.contains("match_name")].unique()

    def _skip(self, button: ipyw.Button | None) -> None:
        """Skip to the next item in the annotation list.

        Parameters
        ----------
        button : type
            Description of button.
        """
        self._index = self._index + 1
        if self._index >= len(self._possible_nodes):
            self._stop(None)
        else:
            buttons = self._initiate_buttons()
            for name in self._possible_names():
                self._add_button(buttons, name)
            self._show_data(buttons)

    def _save_result(self, button: ipyw.Button) -> None:
        """Save the selected match result and skip to next item.

        Parameters
        ----------
        button : type
            Description of button.
        """
        self._annotated_results[
            self._results_data.loc[self._possible_nodes[self._index], "original_name"]
        ] = button.description
        self._skip(None)

    def _add_button(self, buttons: list, name: str) -> None:
        """Add a match name button to the list of buttons.

        Parameters
        ----------
        buttons : type
            Description of buttons.
        name : type
            Description of name.
        """
        button = ipyw.Button(description=name, layout=ipyw.Layout(width="100%"))
        button.on_click(self._save_result)
        buttons.insert(1, button)

    def _show_data(self, buttons: list) -> None:
        """Display the current item and annotation buttons.

        Parameters
        ----------
        buttons : type
            Description of buttons.
        """
        clear_output()
        label = ipyw.Label(
            value=f"{self._results_data.loc[self._possible_nodes[self._index],'original_name']}",
            layout=ipyw.Layout(
                padding="10px",
                font_weight="bold",
                font_size="20px",
                width="100%",
                height="50px",
            ),
        )
        buttons.insert(0, label)
        display(ipyw.VBox(children=buttons, layout=ipyw.Layout(width="500px")))

    def _stop(self, button: ipyw.Button) -> None:
        """Stop the annotation process and clear output.

        Parameters
        ----------
        button : type
            Description of button.
        """
        clear_output()

    def _define_possible_nodes(self, selection) -> list:
        """Define the list of items to annotate based on selection.

        Parameters
        ----------
        selection : type
            Description of selection.

        Returns
        -------
        list : type
            Description of return value.
        """

        if selection == "random":
            return (
                self._results_data.index.to_series()
                .sample(n=len(self._results_data))
                .to_list()
            )

    def start(self, selection: str = "random", fresh_start: bool = False) -> None:
        """Start the annotation process with optional fresh start.

        Parameters
        ----------
        selection : type
            Description of selection.
        fresh_start : type
            Description of fresh_start.
        """
        self._index = -1
        self._possible_nodes = self._define_possible_nodes(selection)
        if fresh_start:
            self._annotated_results = {}
        self._skip(None)
