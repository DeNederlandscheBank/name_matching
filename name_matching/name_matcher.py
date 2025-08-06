import os
import pickle
import re
import warnings
import numpy as np
import pandas as pd
import importlib.resources as importlib_resource
from tqdm import tqdm
from operator import iconcat
from functools import reduce
from unicodedata import category, normalize
from re import escape, sub
from name_matching.data.transliterations import TRANSLITERATION_MAP
from typing import List, Optional, Union, Tuple
from itertools import compress
from sklearn.feature_extraction.text import TfidfVectorizer
from name_matching.distance_metrics import make_distance_metrics
from name_matching.sparse_cosine import sparse_cosine_top_n


class NameMatcher:
    """
    Name matching using character n-gram cosine similarity followed by fuzzy matching.

    This class first vectorizes names using character n-grams (via TF-IDF and cosine
    similarity), selects the top N candidates, and then applies multiple fuzzy matching
    distance metrics to pick the best match or top-K matches.

    Parameters
    ----------
    ngrams : tuple of int, default=(2, 3)
        Character n-gram lengths used for cosine similarity.
    top_n : int, default=50
        Number of top candidates to return from the cosine step.
    low_memory : bool, default=False
        If True, uses a low-memory approach for sparse cosine similarity.
    number_of_rows : int, default=5000
        Batch size for low-memory sparse cosine similarity. Ignored if low_memory is True.
    number_of_matches : int, default=1
        Number of fuzzy-matched alternatives to return.
    lowercase : bool, default=True
        If True, converts text to lowercase during preprocessing.
    non_word_characters : Optional[bool], default=True
        If True, strips non-word characters (excluding & and #) during preprocessing.
    remove_ascii : bool, default=True
        If True, transliterates to ASCII (dropping accents) during preprocessing.
    punctuations : Optional[bool], default=None
        Deprecated alias for non_word_characters.
    legal_suffixes : bool, default=False
        If True, post-processing will ignore common company legal suffixes in scoring.
    preprocess_legal : bool, default=False
        If True, strips or abbreviates legal suffixes/prefixes during preprocessing.
    delete_legal : bool, default=False
        If True, deletes legal suffixes/prefixes instead of abbreviating them.
    make_abbreviations : bool, default=True
        If True, replaces common words with their abbreviations during preprocessing.
    common_words : Union[bool, list], default=False
        If True, will post-process to down-weight the most common words. If a list is
        provided, those specific words will be down-weighted.
    cut_off_no_scoring_words : float, default=0.01
        Threshold (fraction of max frequency) above which a word is considered too common.
    preprocess_split : bool, default=False
        If True, performs an additional “split” variant of preprocessing for searching.
    begin_end_legal_pre_suffix : bool, default=True
        If True, only abbreviate legal terms at the beginning or end of names.
    verbose : bool, default=True
        If True, prints progress via tqdm.
    distance_metrics : list of str, default=[
        "overlap", "weighted_jaccard", "ratcliff_obershelp",
        "fuzzy_wuzzy_token_sort", "editex"]
        List of distance metric names to use in the fuzzy-matching step.
    row_numbers : bool, default=False
        If True, returns original DataFrame index values in the match results.
    return_algorithms_score : bool, default=False
        If True, return the full per-algorithm score matrix instead of just combined scores.
    save_intermediate_results : bool, default=False
        If True, saves intermediate pickle files for matching_data, to_be_matched, possible_matches.
    load_intermediate_results : bool, default=False
        If True, attempts to load intermediate pickle files before recomputing.
    intermediate_results_name : dict of str to str, default={
        "matching_data": "df_matching_data_name",
        "to_be_matched": "to_be_matched_name",
        "possible_matches": "possible_matches_name"
    }
        Filenames (without “.pkl”) for saving/loading intermediate results.

    Raises
    ------
    TypeError
        If `common_words` is not a bool or iterable of strings.
    """

    def __init__(
        self,
        ngrams: tuple = (2, 3),
        top_n: int = 50,
        low_memory: bool = False,
        number_of_rows: int = 5000,
        number_of_matches: int = 1,
        lowercase: bool = True,
        non_word_characters: bool = None,
        remove_ascii: bool = True,
        punctuations: bool = None,
        legal_suffixes: bool = False,
        preprocess_legal: bool = False,
        delete_legal: bool = False,
        make_abbreviations: bool = True,
        common_words: Union[bool, list] = False,
        cut_off_no_scoring_words: float = 0.01,
        preprocess_split: bool = False,
        begin_end_legal_pre_suffix: bool = True,
        verbose: bool = True,
        distance_metrics: Union[list, tuple] = [
            "overlap",
            "weighted_jaccard",
            "ratcliff_obershelp",
            "fuzzy_wuzzy_token_sort",
            "editex",
        ],
        row_numbers: bool = False,
        return_algorithms_score: bool = False,
        save_intermediate_results: bool = False,
        load_intermediate_results: bool = False,
        intermediate_results_name: dict[str, str] = {
            "matching_data": "df_matching_data_name",
            "to_be_matched": "to_be_matched_name",
            "possible_matches": "possible_matches_name",
        },
    ):

        self._possible_matches = None
        self._preprocessed = False
        self._df_matching_data = pd.DataFrame()

        self._number_of_rows = number_of_rows
        self._low_memory = low_memory
        self._save = save_intermediate_results
        self._load = load_intermediate_results
        self._intermediate_results_name = intermediate_results_name

        self._column = ""
        self._column_matching = ""

        self._verbose = verbose
        self._number_of_matches = number_of_matches
        self._top_n = top_n
        self._return_algorithms_score = return_algorithms_score

        self._preprocess_lowercase = lowercase
        self._preprocess_non_word_characters = self._process_punctuations_depricated(
            non_word_characters, punctuations
        )
        self._preprocess_ascii = remove_ascii
        self._preprocess_abbreviations = make_abbreviations
        self._preprocess_legal_suffixes = preprocess_legal
        self._postprocess_company_legal_id = legal_suffixes
        self._delete_legal = delete_legal

        if isinstance(common_words, bool):
            self._postprocess_common_words = common_words
            self._word_set = set()
        elif isinstance(common_words, (list, tuple, set)):
            self._postprocess_common_words = False
            self._word_set = set(common_words)
        else:
            raise TypeError("Please provide common_words as a list or a bool")

        self._preprocess_split = preprocess_split
        self._cut_off = cut_off_no_scoring_words

        if self._postprocess_company_legal_id:
            self._word_set = self._make_no_scoring_words(
                "legal", self._word_set, self._cut_off
            )

        self._original_indexes = not row_numbers
        self._original_index = None
        self._begin_end_legal_pre_suffix = begin_end_legal_pre_suffix

        self.set_distance_metrics(distance_metrics)  # type: ignore

        self._vec = TfidfVectorizer(
            lowercase=False, analyzer="char", ngram_range=(ngrams)
        )
        self._n_grams_matching = None

    def _process_punctuations_depricated(
        self, non_word_characters: bool, punctuations: bool
    ) -> bool:
        """
        Handle backward compatibility between `punctuations` and `non_word_characters`.

        Parameters
        ----------
        non_word_characters : bool
            New parameter indicating whether to strip non-word characters.
        punctuations : bool
            Deprecated parameter (alias of `non_word_characters`).

        Returns
        -------
        bool
            Final value to use for stripping non-word characters.

        Warnings
        --------
        Warns if both parameters are provided but disagree.
        """

        if non_word_characters is None:
            if punctuations is None:
                return True
            else:
                warnings.warn(
                    "The punctuations parameter has been replaced with the non_word_characters "
                    + "parameter, please use the non_word_characters parameter going forward. The"
                    + "punctuations parameter will be depricated in the future."
                )
                return punctuations
        else:
            if (punctuations is not None) & (punctuations != non_word_characters):
                warnings.warn(
                    "non_word_characters is the new name of the punctuations parameter. "
                    + "These parameters are now not equal and the non_word_characters is used."
                    + "The punctuations parameter will be depricated in the future."
                )
            return non_word_characters

    def _generate_combinations(
        self,
        list_a: List,
        list_b: List,
        ind: int = 0,
        result: Optional[List] = None,
    ) -> None:
        """
        Recursively build all possible element-wise choices between two lists.

        At each index `i`, you may choose `list_a[i]` or `list_b[i]`.

        Parameters
        ----------
        list_a : list
            First choice list.
        list_b : list
            Second choice list.
        ind : int, default=0
            Current index in recursion.
        result : list, optional
            Sequence built so far.

        Returns
        -------
        None
            Results are appended to `self._temp`.
        """
        if result is None:
            result = []

        if ind == len(list_a):
            self._temp.append(result)
            return

        self._generate_combinations(list_a, list_b, ind + 1, result + [list_a[ind]])
        self._generate_combinations(list_b, list_a, ind + 1, result + [list_b[ind]])

    def _replace_substring(
        self,
        name: str,
        abbreviations: List[str],
        long_names: List[str],
        begin_end: bool = True,
        delete_names: bool = False,
    ) -> str:
        """
        Replace or delete substrings in `name` according to provided maps.

        Parameters
        ----------
        name : str
            Original string.
        abbreviations : list of str
            Short forms to insert.
        long_names : list of str
            Corresponding full forms to replace.
        begin_end : bool, default=True
            If True, only replace when the full form is at the start or end.
        delete_names : bool, default=False
            If True, delete the matched substring entirely instead of abbreviating.

        Returns
        -------
        str
            Modified string.
        """
        if begin_end:
            for abbreviation, long_name in zip(abbreviations, long_names):
                if name.startswith(long_name) | name.endswith(long_name):
                    if delete_names:
                        name = re.sub(rf"\b{long_name}$", "", name)
                        name = re.sub(rf"^{long_name}\b", "", name)
                    else:
                        name = re.sub(rf"\b{long_name}$", abbreviation, name)
                        name = re.sub(rf"^{long_name}\b", abbreviation, name)
        else:
            for abbreviation, long_name in zip(abbreviations, long_names):
                if long_name in name:
                    if delete_names:
                        name = re.sub(rf"\b{long_name}\b", "", name)
                    else:
                        name = re.sub(rf"\b{long_name}\b", abbreviation, name)

        return name

    def _replace_common_strings(
        self, data: pd.DataFrame, column_name: str
    ) -> pd.DataFrame:
        """
        Abbreviate common words in a DataFrame column using an external CSV map.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame to modify.
        column_name : str
            Column on which to apply replacements.

        Returns
        -------
        pd.DataFrame
            Modified DataFrame.
        """
        with importlib_resource.as_file(
            importlib_resource.files("name_matching.data").joinpath("common_words.csv")
        ) as path:
            common_words = pd.read_csv(path)
        data.loc[:, column_name] = data.loc[:, column_name].apply(
            lambda x: self._replace_substring(
                x,
                common_words["short_form"].tolist(),
                common_words["word"].tolist(),
                begin_end=False,
            )
        )

        return data

    def _replace_legal_pre_suffixes_with_abbreviations(
        self, data: pd.DataFrame, column_name: str
    ) -> pd.DataFrame:
        """
        Abbreviate or delete legal prefixes/suffixes in a DataFrame column.
        Reads `legal_names.csv` to get mappings.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing the column.
        column_name : str
            Name of the column to process.

        Returns
        -------
        pd.DataFrame
            DataFrame with legal terms replaced.
        """
        abbreviations = []
        possible_names = []
        with importlib_resource.as_file(
            importlib_resource.files("name_matching.data").joinpath("legal_names.csv")
        ) as path:
            legal_words = pd.read_csv(path)

        for _, legal_word in legal_words.iterrows():
            abbr = re.split(r"[. /]", legal_word["abbreviation"].strip().lower())
            abbr = list(filter(None, abbr))
            lgl = legal_word["full_name"].lower().strip().split(" ")

            if len(abbr) == len(lgl):
                self._temp = []
                self._generate_combinations(abbr, lgl)
            elif len(abbr) < len(lgl):
                new_lgl = self._combine_legal_words(abbr, lgl)
                if len(new_lgl) == len(abbr):
                    self._temp = []
                    self._generate_combinations(abbr, new_lgl)
                else:
                    self._temp = [legal_word["full_name"]]
            else:
                self._temp = [legal_word["full_name"]]

            self._temp.append("".join(abbr))

            for option in self._temp:
                abbreviations.append(legal_word["abbreviation"].lower())
                if self._preprocess_non_word_characters:
                    possible_names.append(
                        option.strip()
                        if isinstance(option, str)
                        else " ".join(option).strip()
                    )
                else:
                    if isinstance(option, str):
                        possible_names.append(option.strip())
                    else:
                        possible_names.append(" ".join(option).strip())
                        possible_names.append(".".join(option).strip() + ".")
                        abbreviations.append(legal_word["abbreviation"].lower())

        if self._delete_legal:
            possible_names.sort(key=len, reverse=True)

        data[column_name] = data.apply(
            lambda x: self._replace_substring(
                x[column_name],
                abbreviations,
                possible_names,
                begin_end=self._begin_end_legal_pre_suffix,
                delete_names=self._delete_legal,
            ),
            axis=1,
        )

        return data

    def _combine_legal_words(self, abbr: List[str], lgl: List[str]) -> List[str]:
        """
        Merge parts of a full legal name so they align with its abbreviation parts.

        Parameters
        ----------
        abbr : list of str
            Segments of the abbreviation.
        lgl : list of str
            Segments of the full legal name.

        Returns
        -------
        list of str
            Re-grouped segments of the legal name.
        """
        ind = 0
        new_lgl = []
        combined_name = ""
        for letter in abbr:
            while ind < len(lgl) and not lgl[ind].startswith(letter):
                combined_name += " " + lgl[ind]
                ind += 1
            if ind < len(lgl) and lgl[ind].startswith(letter):
                if combined_name:
                    new_lgl.append(combined_name.strip())
                combined_name = lgl[ind]
                ind += 1
        if combined_name:
            new_lgl.append(combined_name.strip())
        return new_lgl

    def set_distance_metrics(self, metrics: List) -> None:
        """
        A method to set which of the distance metrics should be employed during the
        fuzzy matching. For very short explanations of most of the name matching
        algorithms please see the make_distance_metrics function in distance_matrics.py

        Parameters
        ----------
        metrics: List
            The list with the distance metrics to be used during the name matching. The
            distance metrics can be chosen from the list below:
                indel
                discounted_levenshtein
                tichy
                cormodeL_z
                iterative_sub_string
                baulieu_xiii
                clement
                dice_asymmetricI
                kuhns_iii
                overlap
                pearson_ii
                weighted_jaccard
                warrens_iv
                bag
                rouge_l
                q_grams
                ratcliff_obershelp
                ncd_bz2
                fuzzy_wuzzy_partial_string
                fuzzy_wuzzy_token_sort
                fuzzy_wuzzy_token_set
                editex
                typo
                lig_3
                ssk
                refined_soundex
                double_metaphone
        """

        input_metrics = {str(metric).lower(): True for metric in metrics}
        try:
            self._distance_metrics = make_distance_metrics(**input_metrics)
        except TypeError:
            raise TypeError(
                "Not all of the supplied distance metrics are available. Please check the"
                + "list of options in the make_distance_metrics function and adjust"
                + " your list accordingly"
            )
        self._num_distance_metrics = sum(
            [len(x) for x in self._distance_metrics.values()]
        )

    def _select_top_words(
        self, word: str, word_counts: pd.Series, occurrence_count: int
    ) -> str:
        """Remove the top words from the string word based on an occurrence_count threshold

        Parameters
        ----------
        word: str
            the string from which the words should be removed
        word_counts: pd.Series
            the words which should be removed with their counts as result from a value_counts
        occurrence_count: int
            the multiplication factor of the minimum occurrences below which to select

        Returns
        -------
        str
           The string word with the words with a too high word_counts removed
        """
        compressed_list = list(
            compress(
                word,
                (word_counts[word] < occurrence_count * word_counts[word].min()).values,
            )
        )

        return " ".join(compressed_list)

    def _preprocess_reduce(
        self, to_be_matched: pd.DataFrame, occurrence_count: int = 3
    ) -> pd.DataFrame:
        """Preprocesses and copies the data to obtain the data with reduced strings. The
        strings have all words removed which appear more than 3x as often as the least
        common word in the string and returns an adjusted copy of the input

        Parameters
        ----------
        to_be_matched: pd.DataFrame
            A dataframe from which the most common words should be removed
        occurrence_count: int
            The number of occurrence a word can occur more then the least common word in
            the string for which it will still be included in the process
            default=3

        Returns
        -------
        pd.DataFrame
            A dataframe that will contain the reduced strings
        """
        individual_words = (
            to_be_matched[self._column_matching].str.split(expand=True).stack()
        )
        word_counts = individual_words.value_counts()
        to_be_matched_new = to_be_matched.copy()
        name = to_be_matched[self._column_matching].str.split()
        to_be_matched_new[self._column_matching] = name.apply(
            lambda word: self._select_top_words(word, word_counts, occurrence_count)
        )

        return to_be_matched_new

    def load_and_process_master_data(
        self,
        column: str,
        df_matching_data: pd.DataFrame,
        start_processing: bool = True,
        transform: bool = True,
    ) -> None:
        """Load the matching data into the NameMatcher and start the preprocessing.

        Parameters
        ----------
        column : string
            The column name of the dataframe which should be used for the matching
        df_matching_data: pd.DataFrame
            The dataframe which is used to match the data to.
        start_processing : bool
            A boolean indicating whether to start the preprocessing step after
            loading the matching data. If transform is True the data will still be
            transformed and the preprocessing will be marked as completed.
            default: True
        transform : bool
            A boolean indicating whether or not the data should be transformed after
            the vectoriser is initialised
            default: True
        """
        self._column = column
        self._df_matching_data = df_matching_data
        self._original_index = df_matching_data.index
        if start_processing:
            self._process_matching_data(transform)
        elif transform:
            self._vectorise_data(transform)
            self._preprocessed = True

    def _process_matching_data(self, transform: bool = True) -> None:
        """Function to process the matching data. First the matching data is preprocessed
        and assigned to a variable within the NameMatcher. Next the data is used to
        initialise the TfidfVectorizer.

        Parameters
        ----------
        transform : bool
            A boolean indicating whether or not the data should be transformed after the
            vectoriser is initialised
            default: True
        """
        if self._load and os.path.exists(
            f"{self._intermediate_results_name['matching_data']}.pkl"
        ):
            with open(
                f"{self._intermediate_results_name['matching_data']}.pkl", "rb"
            ) as file:
                self._n_grams_matching = pickle.load(file).fillna("na")
        else:
            self._df_matching_data = self.preprocess(
                self._df_matching_data, self._column
            )

        if self._save:
            with open(
                f"{self._intermediate_results_name['matching_data']}.pkl", "wb"
            ) as file:
                pickle.dump(self._df_matching_data, file)

        if self._postprocess_common_words:
            self._word_set = self._make_no_scoring_words(
                "common", self._word_set, self._cut_off
            )

        self._vectorise_data(transform)
        self._preprocessed = True

    def match_names(
        self, to_be_matched: Union[pd.Series, pd.DataFrame], column_matching: str
    ) -> Union[pd.Series, pd.DataFrame] | Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Match input names against the preprocessed master data.

        This will:
          1. Preprocess the new names.
          2. Compute cosine-similarity top-N candidates.
          3. Apply fuzzy matching to those candidates.

        Parameters
        ----------
        to_be_matched : pandas Series or DataFrame
            New names to match.
        column_matching : str
            Column name in `to_be_matched` containing the names.

        Returns
        -------
        pandas Series or DataFrame
            If `return_algorithms_score=False` and `number_of_matches=1`, returns a
            DataFrame containing:
              - original_name
              - match_name
              - score
              - match_index

            If `return_algorithms_score=False` and `number_of_matches>1`, returns a
            DataFrame with columns for each alternative.

            If `return_algorithms_score=True`, returns a tuple (
            DataFrame_of_scores, DataFrame_of_matched_names ).
        """
        if self._column == "":
            raise ValueError(
                "Please first load the master data via the method: "
                + "load_and_process_master_data"
            )
        if self._verbose:
            tqdm.pandas()
            tqdm.write("preprocessing...\n")
        self._column_matching = column_matching

        is_dataframe = True
        if isinstance(to_be_matched, pd.Series):
            is_dataframe = False
            to_be_matched = pd.DataFrame(
                [to_be_matched.values], columns=to_be_matched.index.to_list()
            )

        if self._load and os.path.exists(
            f"{self._intermediate_results_name['to_be_matched']}.pkl"
        ):
            with open(
                f"{self._intermediate_results_name['to_be_matched']}.pkl", "rb"
            ) as file:
                to_be_matched = pickle.load(file).fillna("na")
        else:
            to_be_matched = self.preprocess(to_be_matched, self._column_matching)

        if self._save:
            with open(
                f"{self._intermediate_results_name['to_be_matched']}.pkl", "wb"
            ) as file:
                pickle.dump(to_be_matched, file)

        if self._load and os.path.exists(
            f"{self._intermediate_results_name['possible_matches']}.pkl"
        ):
            with open(
                f"{self._intermediate_results_name['possible_matches']}.pkl", "rb"
            ) as file:
                self._possible_matches = pickle.load(file)

            if not self._preprocessed:
                self._process_matching_data(False)

        else:
            if not self._preprocessed:
                self._process_matching_data()

            to_be_matched = self.preprocess(to_be_matched, self._column_matching)  # type: ignore
            if self._verbose:
                tqdm.write("preprocessing complete \n searching for matches...\n")
            self._possible_matches = self._search_for_possible_matches(to_be_matched)  # type: ignore

        if self._save:
            with open(
                f"{self._intermediate_results_name['possible_matches']}.pkl", "wb"
            ) as file:
                pickle.dump(self._possible_matches, file)

        if self._preprocess_split:
            self._possible_matches = np.hstack(
                (
                    self._search_for_possible_matches(
                        self._preprocess_reduce(to_be_matched)  # type: ignore
                    ),
                    self._possible_matches,
                )
            )

        if self._verbose:
            tqdm.write("possible matches found   \n fuzzy matching...\n")
            data_matches = to_be_matched.progress_apply(
                lambda x: self.fuzzy_matches(
                    self._possible_matches[to_be_matched.index.get_loc(x.name), :], x  # type: ignore
                ),
                axis=1,
            )  # type: ignore
        else:
            data_matches = to_be_matched.apply(
                lambda x: self.fuzzy_matches(
                    self._possible_matches[to_be_matched.index.get_loc(x.name), :], x  # type: ignore
                ),
                axis=1,
            )
        if self._return_algorithms_score:
            return data_matches, self._df_matching_data.iloc[
                self._possible_matches.flatten(), :
            ][
                self._column
            ].values.reshape(  # type: ignore
                (-1, self._top_n)
            )

        if self._number_of_matches == 1:
            data_matches = data_matches.rename(
                columns={
                    "match_name_0": "match_name",
                    "score_0": "score",
                    "match_index_0": "match_index",
                }
            )
        if is_dataframe and self._original_indexes:
            for col in data_matches.columns[
                data_matches.columns.str.contains("match_index")
            ]:
                data_matches[col] = self._original_index[  # type: ignore
                    data_matches[col].astype(int).fillna(0)
                ]  # type: ignore

        if self._verbose:
            tqdm.write("done")

        return data_matches

    def fuzzy_matches(
        self, possible_matches: np.array, to_be_matched: pd.Series  # type: ignore
    ) -> pd.Series:
        """A method which performs the fuzzy matching between the data in the
        to_be_matched series as well as the indicated indexes of the matching_data points
        which are possible matching candidates.

        Parameters
        ----------
        possible_matches : np.array
            An array containing the indexes of the matching data with potential matches
        to_be_matched : pd.Series
            The data which should be matched

        Returns
        -------
        pd.Series
            A series containing the match index from the matching_data dataframe. the name
            in the to_be_matched data, the name to which the datapoint was matched and a
            score between 0 (no match) and 100(perfect match) to indicate the quality of
            the matches.
        """
        if len(possible_matches.shape) > 1:
            possible_matches = possible_matches[0]

        indexes = np.array(
            [
                [f"match_name_{num}", f"score_{num}", f"match_index_{num}"]
                for num in range(self._number_of_matches)
            ]
        ).flatten()
        match = pd.Series(index=np.append("original_name", indexes), dtype=object)
        match["original_name"] = to_be_matched[self._column_matching]
        list_possible_matches = self._df_matching_data.iloc[
            possible_matches.flatten(), :
        ][self._column].values

        match_score = self._score_matches(
            to_be_matched[self._column_matching], list_possible_matches
        )
        if self._return_algorithms_score:
            return match_score
        ind = self._rate_matches(match_score)

        for num, col_num in enumerate(ind):
            match[f"match_name_{num}"] = list_possible_matches[col_num]
            match[f"match_index_{num}"] = possible_matches[col_num]

        match = self._adjust_scores(match_score[ind, :], match)

        if len(self._word_set):
            match = self.postprocess(match)

        return match

    def _score_matches(
        self, to_be_matched_instance: str, possible_matches: List
    ) -> np.array:  # type: ignore
        """A method to score a name to_be_matched_instance to a list of possible matches.
        The scoring is done based on all the metrics which are enabled.

        Parameters
        ----------
        to_be_matched_instance : str
            The name which should match one of the possible matches
        possible_matches : List
            list of the names of the possible matches

        Returns
        -------
        np.array
            The score of each of the matches with respect to the different metrics which
            are assessed.
        """
        match_score = np.zeros((len(possible_matches), self._num_distance_metrics))
        idx = 0
        for method_list in self._distance_metrics.values():
            for method in method_list:
                match_score[:, idx] = np.array(
                    [
                        method.sim(str(to_be_matched_instance), str(s))
                        for s in possible_matches
                    ]
                )
                idx = idx + 1

        return match_score

    def _rate_matches(self, match_score: np.array) -> np.array:  # type: ignore
        """Converts the match scores from the score_matches method to a list of indexes of
        the best scoring matches limited to the _number_of_matches.

        Parameters
        ----------
        match_score : np.array
            An array containing the scores of each of the possible alternatives for each
            of the different methods used

        Returns
        -------
        np.array
            The indexes of the best rated matches
        """
        if self._number_of_matches == 1:
            ind = [np.argmax(np.mean(match_score, axis=1))]
        elif self._number_of_matches == len(self._distance_metrics):
            ind = np.zeros(len(self._distance_metrics))
            idx = 0
            for num, method_list in enumerate(self._distance_metrics.values()):
                method_grouped_results = np.reshape(
                    match_score[:, idx : idx + len(method_list)], (-1, len(method_list))
                )
                ind[num] = np.argmax(np.mean(method_grouped_results, axis=1))
                idx = idx + len(method_list)
        elif self._number_of_matches == self._num_distance_metrics:
            ind = np.argmax(match_score, axis=0).reshape(-1)
        else:
            ind = np.argsort(np.mean(match_score, axis=1))[-self._number_of_matches :][
                ::-1
            ]

        return np.array(ind, dtype=int)

    def _get_alternative_names(self, match: pd.Series) -> List:
        """Gets all the possible match names from the match.

        Parameters
        ----------
        match : pd.Series
            The series with the possible matches and original scores

        Returns
        -------
        list
            A list with the alternative names for this match
        """
        alt_names = []

        for num in range(self._number_of_matches):
            alt_names.append(str(match[f"match_name_{num}"]))

        return alt_names

    def _process_words(self, org_name: str, alt_names: List) -> Tuple[str, List]:
        """Removes the words from the word list from the org_name and all the names in
        alt_names .

        Parameters
        ----------
        org_name : str
            The original name for the matching data
        alt_names : list
            A list of names from which the words should be removed

        Returns
        -------
        Tuple[str, list]
            The processed version of the org_name and the alt_names, with the words
            removed
        """
        len_atl_names = len(alt_names)
        for word in self._word_set:
            org_name = " ".join(sub(rf"\b{escape(word)}\b", "", org_name).split())
            for num in range(len_atl_names):
                alt_names[num] = " ".join(
                    sub(rf"\b{escape(word)}\b", "", alt_names[num]).split()
                )

        return org_name, alt_names

    def _adjust_scores(self, match_score: np.array, match: pd.Series) -> pd.Series:  # type: ignore
        """Adjust the scores to be between 0 and 100

        Parameters
        ----------
        match_score : np.array
            An array with the scores for each of the options
        match : pd.Series
            The series with the possible matches and original scores

        Returns
        -------
        pd.Series
            The series with the possible matches and adjusted scores
        """
        for num in range(self._number_of_matches):
            match[f"score_{num}"] = 100 * np.mean(match_score[num, :])

        return match

    def postprocess(self, match: pd.Series) -> pd.Series:
        """Postprocesses the scores to exclude certain specific company words or the
        most common words. In this method only the scores are adjusted, the matches
        still stand.

        Parameters
        ----------
        match : pd.Series
            The series with the possible matches and original scores

        Returns
        -------
        pd.Series
            A new version of the input series with updated scores
        """
        alt_names = self._get_alternative_names(match)
        org_name = str(match["original_name"])

        org_name, alt_names = self._process_words(org_name, alt_names)

        match_score = self._score_matches(org_name, alt_names)
        ind = self._rate_matches(match_score)

        match = self._adjust_scores(match_score[ind, :], match)

        return match

    def _vectorise_data(self, transform: bool = True) -> None:
        """Initialises the TfidfVectorizer, which generates ngrams and weights them
        based on the occurrance. Subsequently the matching data will be used to fit
        the vectoriser and the matching data might also be send to the transform_data
        function depending on the transform boolean.

        Parameters
        ----------
        transform : bool
            A boolean indicating whether or not the data should be transformed after the
            vectoriser is initialised
            default: True
        """
        self._vec.fit(self._df_matching_data[self._column].values.flatten().astype(str))  # type: ignore
        if transform:
            self.transform_data()

    def transform_data(self) -> None:
        """A method which transforms the matching data based on the ngrams transformer.
        After the transformation (the generation of the ngrams), the data is normalised
        by dividing each row by the sum of the row. Subsequently the data is changed to
        a coo sparse matrix format with the column indices in ascending order.
        """
        ngrams = self._vec.transform(self._df_matching_data[self._column].astype(str))
        for i, j in zip(ngrams.indptr[:-1], ngrams.indptr[1:]):
            ngrams.data[i:j] = ngrams.data[i:j] / np.sum(ngrams.data[i:j])
        self._n_grams_matching = ngrams.tocsc()
        if self._low_memory:
            self._n_grams_matching = self._n_grams_matching.tocoo()

    def _search_for_possible_matches(self, to_be_matched: pd.DataFrame) -> np.array:  # type: ignore
        """Generates ngrams from the data which should be matched, calculate the cosine
        simularity between these data and the matching data. Hereafter a top n of the
        matches is selected and returned.

        Parameters
        ----------
        to_be_matched : pd.Series
            A series containing a single instance of the data to be matched

        Returns
        -------
        np.array
            An array of top n values which are most closely matched to the to be matched
            data based on the ngrams
        """
        if self._n_grams_matching is None:
            raise RuntimeError(
                """First the data needs to be transformed to be able to use the sparse """
                + """cosine simularity. To transform the data, run transform_data"""
                + """ or run load_and_process_master_data with transform=True"""
            )

        if self._low_memory:
            results = np.zeros((len(to_be_matched), self._top_n))
            input_data = to_be_matched[self._column_matching]
            for idx, row_name in enumerate(tqdm(input_data, disable=not self._verbose)):
                match_ngrams = self._vec.transform([str(row_name)])
                results[idx, :] = sparse_cosine_top_n(
                    matrix_a=self._n_grams_matching,
                    matrix_b=match_ngrams,
                    top_n=self._top_n,
                    low_memory=self._low_memory,
                    number_of_rows=self._number_of_rows,
                    verbose=self._verbose,
                )
        else:
            match_ngrams = self._vec.transform(
                to_be_matched[self._column_matching].astype(str).tolist()
            ).tocsc()
            results = sparse_cosine_top_n(
                matrix_a=self._n_grams_matching,
                matrix_b=match_ngrams,
                top_n=self._top_n,
                low_memory=self._low_memory,
                number_of_rows=self._number_of_rows,
                verbose=self._verbose,
            )

        return results

    def unicode_to_ascii(self, text: str) -> str:
        """Converts a string to ascii characters trhough transliteration. The
        transliteration map is stored in the transliterations.py file in the
        data folder.

        Parameters
        ----------
        test : str
            The text to be transliterated to ascii characters

        Returns
        -------
        str
            The process text without any non-ascii characters
        """

        normalized_text = normalize("NFD", text)

        return (
            "".join(
                [
                    TRANSLITERATION_MAP.get(char, char)
                    for char in normalized_text
                    if category(char) != "Mn"
                ]
            )
            .encode("ASCII", "ignore")
            .decode()
        )

    def preprocess(
        self, df: pd.DataFrame, column_name: str, original_name: bool = False
    ) -> pd.DataFrame:
        """Preprocess a dataframe before applying a name matching algorithm. The
        preprocessing consists of removing special characters, spaces, converting all
        characters to lower case and removing the words given in the word lists

        Parameters
        ----------
        df : DataFrame
            The dataframe or series on which the preprocessing needs to be performed
        column_name : str
            The name of the column that is used for the preprocessing
        original_name : bool
            If True, returns an additional column 'original_name' in the dataframe
            this column holds the original, non-processed name.
            default=False

        Returns
        -------
        pd.DataFrame
            The preprocessed dataframe or series depending on the input
        """
        if original_name:
            df["original_name"] = df[column_name].copy(deep=True)
        df.loc[:, column_name] = df[column_name].astype(str)
        if self._preprocess_non_word_characters:
            df.loc[:, column_name] = df[column_name].str.replace(
                r"[^\w\-\&\#]", " ", regex=True
            )
            df.loc[:, column_name] = (
                df[column_name].str.replace(r"\s+", " ", regex=True).str.strip()
            )
        if self._preprocess_ascii:
            df.loc[:, column_name] = df[column_name].apply(
                lambda string: self.unicode_to_ascii(str(string))
            )
        if self._preprocess_lowercase:
            df.loc[:, column_name] = df[column_name].str.lower()
        if self._preprocess_legal_suffixes:
            df = self._replace_legal_pre_suffixes_with_abbreviations(df, column_name)
        if self._preprocess_abbreviations:
            df = self._replace_common_strings(df, column_name)
        if self._preprocess_non_word_characters:
            df.loc[:, column_name] = df[column_name].str.replace(
                r"[^\w\-\&\#]", " ", regex=True
            )
            df.loc[:, column_name] = (
                df[column_name].str.replace(r"\s+", " ", regex=True).str.strip()
            )

        return df

    def _preprocess_word_list(self, terms: dict) -> List:
        """Preprocess legal words to remove non-word-characters and trailing leading space

        Parameters
        -------
        terms: dict
            a dictionary of legal words

        Returns
        -------
        list
            A list of preprocessed legal words
        """
        if self._preprocess_non_word_characters:
            return [
                sub(r"[^\w\-\&\#]", "", s).strip()
                for s in reduce(iconcat, terms.values(), [])
            ]
        else:
            return [s.strip() for s in reduce(iconcat, terms.values(), [])]

    def _process_legal_words(self, word_set: set) -> set:
        """Preprocess legal words and add them to the word_set

        Parameters
        -------
        word_set: str
            the current word list which should be extended with additional words

        Returns
        -------
        Set
            The original word_set with the legal words added
        """
        with importlib_resource.as_file(
            importlib_resource.files("name_matching.data").joinpath("legal_names.csv")
        ) as path:
            legal_words = pd.read_csv(path)
        word_set = word_set.union(set(legal_words["abbreviation"].values))

        return word_set

    def _process_common_words(self, word_set: set, cut_off: float) -> set:
        """A method to select the most common words from the matching_data.

        Parameters
        -------
        word_set: str
            the current word list which should be extended with additional words
        cut_off: float
            the cut_off percentage of the occurrence of the most occurring word for
            which words are still included in the no_soring_words set

        Returns
        -------
        Set
            The current word set with the most common words from the matching_data added
        """
        word_counts = (
            self._df_matching_data[self._column]
            .str.split(expand=True)
            .stack()
            .value_counts()
        )
        word_set = word_set.union(
            set(word_counts[word_counts > np.max(word_counts) * cut_off].index)
        )

        return word_set

    def _make_no_scoring_words(
        self, indicator: str, word_set: set, cut_off: float
    ) -> set:
        """A method to make a set of words which are not taken into account when
        scoring matches.

        Parameters
        -------
        indicator: str
            indicator for which types of words should be excluded can be legal for
            legal suffixes or common for the most common words
        word_set: str
            the current word list which should be extended with additional words
        cut_off: float
            the cut_off percentage of the occurrence of the most occurring word for
            which words are still included in the no_soring_words set

        Returns
        -------
        Set
            The set of no scoring words
        """
        if indicator == "legal":
            word_set = self._process_legal_words(word_set)
        if indicator == "common":
            word_set = self._process_common_words(word_set, cut_off)

        return word_set
