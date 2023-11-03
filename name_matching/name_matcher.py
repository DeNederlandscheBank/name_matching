import unicodedata
import functools
import operator
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Union, Tuple
from itertools import compress
from sklearn.feature_extraction.text import TfidfVectorizer
from name_matching.distance_metrics import make_distance_metrics
from cleanco.termdata import terms_by_type, terms_by_country
from name_matching.sparse_cosine import sparse_cosine_top_n


class NameMatcher:

    """
    A class for the name matching of data based on the strings in a single column. The NameMatcher
    first applies a cosine similarity on the ngrams of the strings to get an approximate match followed 
    by a fuzzy matching based on a number of different algorithms.

    Parameters
    ----------
    ngrams : tuple of integers
        The length of the ngrams which should be used for the generation of ngrams for the cosine
        similarity comparison of the possible matches
        default=(2, 3)
    top_n : integer
        The number of possible matches that should be included in the group which will be analysed
        with the fuzzy matching algorithms
        default=50
    low_memory : bool
        Bool indicating if the a low memory approach should be taken in the sparse cosine similarity
        step.
        default=False
    number_of_rows : integer
        Determines how many rows should be calculated at once with the sparse cosine similarity step.
        If the low_memory bool is True this number is unused.
        default=5000
    number_of_matches : int
        The number of matches which should be returned by the matching algorithm. If a number higher
        than 1 is given, a number of alternative matches are also returned. If the number is equal
        to the number of algorithms used, the best match for each algorithm is returned. If the
        number is equal to the number of algorithm groups which are included the best match for each
        group is returned.
        default=1
    legal_suffixes : bool
        Boolean indicating whether the most common company legal terms should be excluded when calculating 
        the final score. The terms are still included in determining the best match.
        default=False
    common_words : bool
        Boolean indicating whether the most common words from the matching data should be excluded 
        when calculating the final score. The terms are still included in determining the best match.
        default=False
    cut_off_no_scoring_words: float
        the cut off percentage of the occurrence of the most occurring word for which words are still included 
        in the no_soring_words set
        default=0.01
    lowercase : bool
        A boolean indicating whether during the preprocessing all characters should be converted to
        lowercase, to generate case insensitive matching
        default=True
    punctuations : bool
        A boolean indicating whether during the preprocessing all punctuations should be ignored
        default=True
    remove_ascii : bool
        A boolean indicating whether during the preprocessing all characters should be converted to
        ascii characters
        default=True : bool
    preprocess_split
        Indicating whether during the preprocessing an additional step should be taken in which only 
        the most common words out of a name are isolated and used in the matching process. The removing 
        of the common words is only done for the n-grams cosine matching part.
        default=False
    verbose : bool
        A boolean indicating whether progress printing should be done
        default=True
    distance_metrics: list
        A list of The distance metrics to be used during the fuzzy matching. For a list of possible distance
        metrics see the distance_metrics.py file. By default the following metrics are used: overlap, weighted_jaccard, 
                ratcliff_obershelp, fuzzy_wuzzy_token_sort and editex.
    row_numbers : bool
        Bool indicating whether the row number should be used as match_index rather than the original index as
        was the default case before version 0.8.8  
        default=False
    """

    def __init__(self,
                 ngrams: tuple = (2, 3),
                 top_n: int = 50,
                 low_memory: bool = False,
                 number_of_rows: int = 5000,
                 number_of_matches: int = 1,
                 lowercase: bool = True,
                 punctuations: bool = True,
                 remove_ascii: bool = True,
                 legal_suffixes: bool = False,
                 common_words: bool = False,
                 cut_off_no_scoring_words: float = 0.01,
                 preprocess_split: bool = False,
                 verbose: bool = True,
                 distance_metrics: Union[list, tuple] = ['overlap', 'weighted_jaccard', 'ratcliff_obershelp',
                                                         'fuzzy_wuzzy_token_sort', 'editex'],
                 row_numbers: bool = False):

        self._possible_matches = None
        self._preprocessed = False
        self._word_set = set()
        self._df_matching_data = pd.DataFrame()

        self._number_of_rows = number_of_rows
        self._low_memory = low_memory

        self._column = ''
        self._column_matching = ''

        self._verbose = verbose
        self._number_of_matches = number_of_matches
        self._top_n = top_n

        self._preprocess_lowercase = lowercase
        self._preprocess_punctuations = punctuations
        self._preprocess_ascii = remove_ascii
        self._postprocess_company_legal_id = legal_suffixes
        self._postprocess_common_words = common_words

        self._preprocess_split = preprocess_split
        self._cut_off = cut_off_no_scoring_words

        if self._postprocess_company_legal_id:
            self._word_set = self._make_no_scoring_words(
                'legal', self._word_set, self._cut_off)
            
        self._original_indexes = not row_numbers
        self._original_index = None

        self.set_distance_metrics(distance_metrics)

        self._vec = TfidfVectorizer(
            lowercase=False, analyzer="char", ngram_range=(ngrams))
        self._n_grams_matching = None

    def set_distance_metrics(self, metrics: list) -> None:
        """
        A method to set which of the distance metrics should be employed during the
        fuzzy matching. For very short explanations of most of the name matching 
        algorithms please see the make_distance_metrics function in distance_matrics.py

        Parameters
        ----------
        metrics: list 
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
            raise TypeError('Not all of the supplied distance metrics are available. Please check the' +
                            'list of options in the make_distance_metrics function and adjust your list accordingly')
        self._num_distance_metrics = sum(
            [len(x) for x in self._distance_metrics.values()])

    def _select_top_words(self, word: str, word_counts: pd.Series, occurrence_count: int) -> str:
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
        compressed_list = list(compress(
            word, (word_counts[word] < occurrence_count*word_counts[word].min()).values))

        return ' '.join(compressed_list)

    def _preprocess_reduce(self,
                           to_be_matched: pd.DataFrame,
                           occurrence_count: int = 3) -> pd.DataFrame:
        """Preprocesses and copies the data to obtain the data with reduced strings. The strings have all words
        removed which appear more than 3x as often as the least common word in the string and returns an adjusted 
        copy of the input 

        Parameters
        ----------
        to_be_matched: pd.DataFrame 
            A dataframe from which the most common words should be removed
        occurrence_count: int
            The number of occurrence a word can occur more then the least common word in the string for which it will
            still be included in the process 
            default=3

        Returns
        -------
        pd.DataFrame
            A dataframe that will contain the reduced strings
        """
        individual_words = to_be_matched[self._column_matching].str.split(
            expand=True).stack()
        word_counts = individual_words.value_counts()
        to_be_matched_new = to_be_matched.copy()
        name = to_be_matched[self._column_matching].str.split()
        to_be_matched_new[self._column_matching] = name.apply(
            lambda word: self._select_top_words(word, word_counts, occurrence_count))

        return to_be_matched_new

    def load_and_process_master_data(self,
                                     column: str,
                                     df_matching_data: pd.DataFrame,
                                     start_processing: bool = True,
                                     transform: bool = True) -> None:
        """Load the matching data into the NameMatcher and start the preprocessing.

        Parameters
        ----------
        column : string
            The column name of the dataframe which should be used for the matching
        df_matching_data: pd.DataFrame
            The dataframe which is used to match the data to.
        start_processing : bool 
            A boolean indicating whether to start the preprocessing step after loading the matching data 
            default: True
        transform : bool 
            A boolean indicating whether or not the data should be transformed after the vectoriser is initialised 
            default: True
        """
        self._column = column
        self._df_matching_data = df_matching_data        
        self._original_index = df_matching_data.index
        if start_processing:
            self._process_matching_data(transform)

    def _process_matching_data(self,
                               transform: bool = True) -> None:
        """Function to process the matching data. First the matching data is preprocessed and assigned to 
        a variable within the NameMatcher. Next the data is used to initialise the TfidfVectorizer. 

        Parameters
        ----------
        transform : bool 
            A boolean indicating whether or not the data should be transformed after the vectoriser is initialised 
            default: True
        """
        self._df_matching_data = self.preprocess(
            self._df_matching_data, self._column)
        if self._postprocess_common_words:
            self._word_set = self._make_no_scoring_words(
                'common', self._word_set, self._cut_off)
        self._vectorise_data(transform)
        self._preprocessed = True

    def match_names(self,
                    to_be_matched: Union[pd.Series, pd.DataFrame],
                    column_matching: str) -> Union[pd.Series, pd.DataFrame]:
        """Performs the name matching operation on the to_be_matched data. First it does the preprocessing of the 
        data to be matched as well as the matching data if this has not been performed. Subsequently based on 
        ngrams a cosine similarity is computed between the matching data and the data to be matched, to the top n
        matches fuzzy matching algorithms are performed to determine the best match and the quality of the match

        Parameters
        ----------
        to_be_matched: Union[pd.Series, pd.DataFrame]
            The data which should be matched
        column_matching: str
            string indicating the column which will be matched

        Returns
        -------
        Union[pd.Series, pd.DataFrame]
            A series or dataframe depending on the input containing the match index from the matching_data dataframe. 
            the name in the to_be_matched data, the name to which the datapoint was matched and a score between 0 
            (no match) and 100(perfect match) to indicate the quality of the matches
        """
        if self._column == '':
            raise ValueError(
                'Please first load the master data via the method: load_and_process_master_data')
        if self._verbose:
            tqdm.pandas()
            tqdm.write('preprocessing...\n')
        self._column_matching = column_matching

        is_dataframe = True
        if isinstance(to_be_matched, pd.Series):
            is_dataframe = False
            to_be_matched = pd.DataFrame(
                [to_be_matched.values], columns=to_be_matched.index.to_list())
        if not self._preprocessed:
            self._process_matching_data()
        to_be_matched = self.preprocess(to_be_matched, self._column_matching)

        if self._verbose:
            tqdm.write('preprocessing complete \n searching for matches...\n')

        self._possible_matches = self._search_for_possible_matches(
            to_be_matched)

        if self._preprocess_split:
            self._possible_matches = np.hstack((self._search_for_possible_matches(
                self._preprocess_reduce(to_be_matched)), self._possible_matches))
        
        if self._verbose:
            tqdm.write('possible matches found   \n fuzzy matching...\n')
            data_matches = to_be_matched.progress_apply(lambda x: self.fuzzy_matches(
                self._possible_matches[to_be_matched.index.get_loc(x.name), :], x), axis=1)
        else:
            data_matches = to_be_matched.apply(lambda x: self.fuzzy_matches(
                self._possible_matches[to_be_matched.index.get_loc(x.name), :], x), axis=1)

        if self._number_of_matches == 1:
            data_matches = data_matches.rename(columns={'match_name_0': 'match_name',
                                                        'score_0': 'score', 'match_index_0': 'match_index'})
        if is_dataframe and self._original_indexes:
            for col in data_matches.columns[data_matches.columns.str.contains('match_index')]:
                data_matches[col] = self._original_index[data_matches[col].astype(int).fillna(0)]

        if self._verbose:
            tqdm.write('done')

        return data_matches

    def fuzzy_matches(self,
                      possible_matches: np.array,
                      to_be_matched: pd.Series) -> pd.Series:
        """ A method which performs the fuzzy matching between the data in the to_be_matched series as well
        as the indicated indexes of the matching_data points which are possible matching candidates.

        Parameters
        ----------
        possible_matches : np.array
            An array containing the indexes of the matching data with potential matches
        to_be_matched : pd.Series
            The data which should be matched

        Returns
        -------
        pd.Series
            A series containing the match index from the matching_data dataframe. the name in the to_be_matched data,
            the name to which the datapoint was matched and a score between 0 (no match) and 100(perfect match) to 
            indicate the quality of the matches
        """
        if len(possible_matches.shape) > 1:
            possible_matches = possible_matches[0]

        indexes = np.array([[f'match_name_{num}', f'score_{num}', f'match_index_{num}']
                            for num in range(self._number_of_matches)]).flatten()
        match = pd.Series(index=np.append('original_name', indexes), dtype=object)
        match['original_name'] = to_be_matched[self._column_matching]
        list_possible_matches = self._df_matching_data.iloc[
            possible_matches.flatten(), :][self._column].values

        match_score = self._score_matches(
            to_be_matched[self._column_matching], list_possible_matches)
        ind = self._rate_matches(match_score)

        for num, col_num in enumerate(ind):
            match[f'match_name_{num}'] = list_possible_matches[col_num]
            match[f'match_index_{num}'] = possible_matches[col_num]

        match = self._adjust_scores(match_score[ind, :], match)

        if self._postprocess_common_words or self._postprocess_company_legal_id:
            match = self.postprocess(match)

        return match

    def _score_matches(self,
                       to_be_matched_instance: str,
                       possible_matches: list) -> np.array:
        """A method to score a name to_be_matched_instance to a list of possible matches. The scoring is done
        based on all the metrics which are enabled.

        Parameters
        ----------
        to_be_matched_instance : str
            The name which should match one of the possible matches
        possible_matches : list
            list of the names of the possible matches

        Returns
        -------
        np.array
            The score of each of the matches with respect to the different metrics which are assessed.
        """
        match_score = np.zeros(
            (len(possible_matches), self._num_distance_metrics))
        idx = 0
        for method_list in self._distance_metrics.values():
            for method in method_list:
                match_score[:, idx] = np.array(
                    [method.sim(to_be_matched_instance, s) for s in possible_matches])
                idx = idx + 1

        return match_score

    def _rate_matches(self,
                      match_score: np.array) -> np.array:
        """Converts the match scores from the score_matches method to a list of indexes of the best scoring 
        matches limited to the _number_of_matches.

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
                    match_score[:, idx: idx + len(method_list)], (-1, len(method_list)))
                ind[num] = np.argmax(np.mean(method_grouped_results, axis=1))
                idx = idx + len(method_list)
        elif self._number_of_matches == self._num_distance_metrics:
            ind = np.argmax(match_score, axis=1)
        else:
            ind = np.argsort(np.mean(match_score, axis=1)
                             )[-self._number_of_matches:][::-1]

        return np.array(ind, dtype=int)

    def _get_alternative_names(self, match: pd.Series) -> list:
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
            alt_names.append(str(match[f'match_name_{num}']))

        return alt_names

    def _process_words(self, org_name: str, alt_names: list) -> Tuple[str, list]:
        """Removes the words from the word list from the org_name and all the names in alt_names .

        Parameters
        ----------
        org_name : str
            The original name for the matching data
        alt_names : list
            A list of names from which the words should be removed

        Returns
        -------
        Tuple[str, list]
            The processed version of the org_name and the alt_names, with the words removed
        """
        len_atl_names = len(alt_names)
        for word in self._word_set:
            org_name = ' '.join(
                re.sub(fr'\b{re.escape(word)}\b', '', org_name).split())
            for num in range(len_atl_names):
                alt_names[num] = ' '.join(
                    re.sub(fr'\b{re.escape(word)}\b', '', alt_names[num]).split())

        return org_name, alt_names

    def _adjust_scores(self, match_score: np.array, match: pd.Series) -> pd.Series:
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
            match[f'score_{num}'] = 100*np.mean(match_score[num, :])

        return match

    def postprocess(self,
                    match: pd.Series) -> pd.Series:
        """Postprocesses the scores to exclude certain specific company words or the most 
        common words. In this method only the scores are adjusted, the matches still stand.

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
        org_name = str(match['original_name'])

        org_name, alt_names = self._process_words(org_name, alt_names)

        match_score = self._score_matches(org_name, alt_names)

        match = self._adjust_scores(match_score, match)

        return match

    def _vectorise_data(self,
                        transform: bool = True):
        """Initialises the TfidfVectorizer, which generates ngrams and weights them based on the occurrance.
        Subsequently the matching data will be used to fit the vectoriser and the matching data might also be send
        to the transform_data function depending on the transform boolean.

        Parameters
        ----------
        transform : bool 
            A boolean indicating whether or not the data should be transformed after the vectoriser is initialised 
            default: True
        """
        self._vec.fit(self._df_matching_data[self._column].values.flatten())
        if transform:
            self.transform_data()

    def transform_data(self):
        """A method which transforms the matching data based on the ngrams transformer. After the 
        transformation (the generation of the ngrams), the data is normalised by dividing each row
        by the sum of the row. Subsequently the data is changed to a coo sparse matrix format with
        the column indices in ascending order.
        """
        ngrams = self._vec.transform(
            self._df_matching_data[self._column].astype(str))
        for i, j in zip(ngrams.indptr[:-1], ngrams.indptr[1:]):
            ngrams.data[i:j] = ngrams.data[i:j]/np.sum(ngrams.data[i:j])
        self._n_grams_matching = ngrams.tocsc()
        if self._low_memory:
            self._n_grams_matching = self._n_grams_matching.tocoo()

    def _search_for_possible_matches(self,
                                     to_be_matched: pd.DataFrame) -> np.array:
        """Generates ngrams from the data which should be matched, calculate the cosine simularity
        between these data and the matching data. Hereafter a top n of the matches is selected and
        returned.

        Parameters
        ----------
        to_be_matched : pd.Series
            A series containing a single instance of the data to be matched

        Returns
        -------
        np.array
            An array of top n values which are most closely matched to the to be matched data based
            on the ngrams
        """
        if self._n_grams_matching is None:
            raise RuntimeError(
                """First the data needs to be transformed to be able to use the sparse cosine simularity. To""" +
                """transform the data, run transform_data or run load_and_process_master_data with transform=True""")

        if self._low_memory:
            results = np.zeros((len(to_be_matched), self._top_n))
            input_data = to_be_matched[self._column_matching]
            for idx, row_name in enumerate(tqdm(input_data, disable=not self._verbose)):
                match_ngrams = self._vec.transform([row_name])
                results[idx, :] = sparse_cosine_top_n(
                    matrix_a=self._n_grams_matching, matrix_b=match_ngrams, top_n=self._top_n, low_memory=self._low_memory, number_of_rows=self._number_of_rows, verbose=self._verbose)
        else:
            match_ngrams = self._vec.transform(
                to_be_matched[self._column_matching].tolist()).tocsc()
            results = sparse_cosine_top_n(
                matrix_a=self._n_grams_matching, matrix_b=match_ngrams, top_n=self._top_n, low_memory=self._low_memory, number_of_rows=self._number_of_rows, verbose=self._verbose)

        return results

    def preprocess(self,
                   df: pd.DataFrame,
                   column_name: str) -> pd.DataFrame:
        """Preprocess a dataframe before applying a name matching algorithm. The preprocessing consists of 
        removing special characters, spaces, converting all characters to lower case and removing the
        words given in the word lists

        Parameters
        ----------
        df : DataFrame
            The dataframe or series on which the preprocessing needs to be performed        
        column_name : str
            The name of the column that is used for the preprocessing

        Returns
        -------
        pd.DataFrame
            The preprocessed dataframe or series depending on the input
        """
        df.loc[:, column_name] = df[column_name].astype(str)
        if self._preprocess_lowercase:
            df.loc[:, column_name] = df[column_name].str.lower()
        if self._preprocess_punctuations:
            df.loc[:, column_name] = df[column_name].str.replace(
                '[^\w\s]', '', regex=True)
            df.loc[:, column_name] = df[column_name].str.replace(
                '  ', ' ').str.strip()
        if self._preprocess_ascii:
            df.loc[:, column_name] = df[column_name].apply(lambda string: unicodedata.normalize(
                'NFKD', str(string)).encode('ASCII', 'ignore').decode())

        return df

    def _preprocess_word_list(self, terms: dict) -> list:
        """Preprocess legal words to remove punctuations and trailing leading space

        Parameters
        -------
        terms: dict
            a dictionary of legal words

        Returns
        -------
        list
            A list of preprocessed legal words  
        """
        if self._preprocess_punctuations:
            return [re.sub(r'[^\w\s]', '', s).strip() for s in functools.reduce(
                operator.iconcat, terms.values(), [])]
        else:
            return [s.strip() for s in functools.reduce(
                    operator.iconcat, terms.values(), [])]

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
        terms_type = self._preprocess_word_list(terms_by_type)
        terms_country = self._preprocess_word_list(terms_by_country)
        word_set = word_set.union(set(terms_country + terms_type))

        return word_set

    def _process_common_words(self, word_set: set, cut_off: float) -> set:
        """A method to select the most common words from the matching_data.

        Parameters
        -------
        word_set: str
            the current word list which should be extended with additional words  
        cut_off: float
            the cut_off percentage of the occurrence of the most occurring word for which words are still included 
            in the no_soring_words set

        Returns
        -------
        Set
            The current word set with the most common words from the matching_data added
        """
        word_counts = self._df_matching_data[self._column].str.split(
            expand=True).stack().value_counts()
        word_set = word_set.union(
            set(word_counts[word_counts > np.max(word_counts)*cut_off].index))

        return word_set

    def _make_no_scoring_words(self,
                               indicator: str,
                               word_set: set,
                               cut_off: float) -> set:
        """A method to make a set of words which are not taken into account when scoring matches.

        Parameters
        -------
        indicator: str
            indicator for which types of words should be excluded can be legal for
            legal suffixes or common for the most common words
        word_set: str
            the current word list which should be extended with additional words  
        cut_off: float
            the cut_off percentage of the occurrence of the most occurring word for which words are still included 
            in the no_soring_words set

        Returns
        -------
        Set
            The set of no scoring words
        """
        if indicator == 'legal':
            word_set = self._process_legal_words(word_set)
        if indicator == 'common':
            word_set = self._process_common_words(word_set, cut_off)

        return word_set
