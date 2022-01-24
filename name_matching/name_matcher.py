import unicodedata
import functools
import operator
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Union
from itertools import compress
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csc_matrix
from name_matching.distance_metrics import make_distance_metrics
from name_matching.term_data import terms_by_type, terms_by_country
from name_matching.sparse_cosine import sparse_cosine_top_n



class NameMatcher:

    """
    A class for the name matching of data based on the strings in a single column. The nameMatcher
    first applies a cosine simularity on the ngrams of the strings to get an aproximate match followed 
    by a fuzzy matching based on a number of different algorithms.

    Parameters
    ----------
    ngrams : tuple of integers
        The length of the ngrams which should be used for the generation of ngrams for the cosine
        simularity comparrison of the possible matches
        default=(2, 3)
    top_n : integer
        The number of possible matches that should be included in the group which will be analysed
        with the fuzzy matching algorithms
        default=50
    memory_usage : integer
        number indicating the memory usage, number between 0 to 10 indicate progressly
        more memory usage, with 0 being the lowest amount of memory usage limit and 10 the highest. The
        memory usage is also dependent on the size of the matching_data reducing this variable will also 
        reduce the memory usage.
        default=7
    number_of_matches : int
        The number of matches which should be returned by the matching algroithm. If a number higher
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
        Boolean indicating whether the most common words from the matching datashould be excluded 
        when calculating the final score. The terms are still included in determining the best match.
        default=False
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
        Indicating whether during the preporcessing an additional step should be taken in which only 
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

    """

    def __init__(self,
                 ngrams=(2, 3),
                 top_n=50,
                 memory_usage=7,
                 number_of_matches=1,
                 lowercase=True,
                 punctuations=True,
                 remove_ascii=True,
                 legal_suffixes=False,
                 common_words=False,
                 preprocess_split=False,
                 verbose=True,
                 distance_metrics=['overlap', 'weighted_jaccard', 'ratcliff_obershelp', 
                        'fuzzy_wuzzy_token_sort', 'editex']):

        self._possible_matches = None
        self._preprocessed = False
        self._word_set = set()
        self._df_matching_data = pd.DataFrame()

        if not 0 <= memory_usage <= 10:
            raise ValueError('memory_usage should be between 0 and 10 inclusive')
        memory_options = [0, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 5000]
        self._low_memory = memory_options[memory_usage]

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

        self._preprocess_common_words = pd.Series()
        self._preprocess_split = preprocess_split

        if self._postprocess_company_legal_id:
            self._word_set = self._make_no_scoring_words(
                'legal', self._word_set)

        self.set_distance_metrics(distance_metrics)

        self._vec = TfidfVectorizer(
            lowercase=False, analyzer="char", ngram_range=(ngrams))
        self._n_grams_matching = None

    def set_distance_metrics(self,
                             metrics: list) -> None:
        """
        A method to set which of the distance metrics should be employed during the
        fuzzy matching
        """
        input_metrics = dict()
        for metric in metrics:
            input_metrics[metric] = True
        try:
            self._distance_metrics = make_distance_metrics(**input_metrics)
        except TypeError:
            raise TypeError("""Not all of the supplied distance metrics are avialable. Please check the
                list of options in the make_distance_metrics function and adjust your list accordingly""")
        self._num_distance_metrics = sum(
            [len(x) for x in self._distance_metrics.values()])

    def _preprocess_reduce(self,
                           to_be_matched: pd.DataFrame,
                           occurence_count=3) -> pd.DataFrame:
        """
        Preprocesses and copies the data to obtain the data with reduced strings. The strings have all words
        removed which appear more then 3x as often as the least common word in the string and returns an adjusted 
        copy of the input 

        Parameters
        ----------
        to_be_matched: pd.DataFrame 
            A dataframe from which the most common words should be removed
        occurence_count: int
            The number of occurence a word can occur more then the least common word in the string for which it will
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
        preprocess_common_words_inst = self._preprocess_common_words.append(
            word_counts).fillna(0)
        preprocess_common_words_inst.groupby(
            preprocess_common_words_inst.index).sum()

        to_be_matched_new = to_be_matched.copy()
        name = to_be_matched[self._column_matching].str.split()
        to_be_matched_new[self._column_matching] = name.apply(lambda x: ' '.join(list(compress(x, (
            preprocess_common_words_inst[x] < occurence_count*preprocess_common_words_inst[x].min()).values))))

        return to_be_matched_new

    def load_and_process_master_data(self,
                                     column: str,
                                     df_matching_data: pd.DataFrame,
                                     start_processing=True,
                                     transform=True) -> None:
        """
        Load the matching data into the nameMatcher and start the preprocessing.

        Parameters
        ----------
    	column : string
            The column name of the dataframe which should be used for the matching
        start_processing : bool 
            A boolean indicating wether to start the preprocessing step after loading the matching data 
            default: True
        transform : bool 
            A boolean indicating wether or not the data should be transformed after the vectoriser is initialised 
            default: True

        """
        self._column = column
        self._df_matching_data = df_matching_data
        if start_processing:
            self._process_matching_data(transform)

    def _process_matching_data(self,
                               transform=True) -> None:
        """
        Function to process the matching data. Frst the matching data is perprocessed and assinged to 
        a variable within the nameMatcher. Next the data is used to initialise the TfidfVectorizer. 

        Parameters
        ----------
        transform : bool 
            A boolean indicating wether or not the data should be transformed after the vectoriser is initialised 
            default: True

        """
        self._df_matching_data = self.preprocess(
            self._df_matching_data, self._column)
        if self._postprocess_common_words:
            self._word_set = self._make_no_scoring_words(
                'common', self._word_set)
        self._vectorise_data(transform)
        self._preprocessed = True

    def do_name_matching(self,
                         to_be_matched: Union[pd.Series, pd.DataFrame],
                         column_matching: str) -> Union[pd.Series, pd.DataFrame]:
        """
        Performs the name matching operation on the to_be_matched data. First it does the preprocessing of the 
        data to be matched as well as the matching data if this has not been performed. Subsequently based on 
        ngrams a cosine simularity is computed between the matching data and the data to be matched, to the top n
        matches fuzzy matching algorithms are performed to determine the best match and the quality of the match

        Parameters
        ----------
        to_be_matched: Union[pd.Series, pd.DataFrame]
            The data which should be matched
        column_matching: str
            string indiccating the column which will be matched

        Returns
        -------
        Union[pd.Series, pd.DataFrame]
            A series or dataframe depending on the input containg the match index from the matching_data dataframe. 
            the name in the to_be_matched data, the name to which the datapoint was matched and a score between 0 
            (no match) and 100(perfect match) to indicate the quality of the matches
        """

        if self._column == '':
                raise ValueError('Please first load the master data via the method: load_and_process_master_data')
        if self._verbose:
            tqdm.pandas()
            tqdm.write('preprocessing...')
            tqdm.write('')
        self._column_matching = column_matching

        if isinstance(to_be_matched, pd.Series):
            to_be_matched = pd.DataFrame(to_be_matched)
        if not self._preprocessed:
            self._process_matching_data()
        to_be_matched = self.preprocess(to_be_matched, self._column_matching)

        if self._verbose:
            tqdm.write('preprocessing complete')
            tqdm.write('searching for matches...')
            tqdm.write('')

        self._possible_matches = self._search_for_possible_matches(
            to_be_matched)

        if self._preprocess_split:
            self._possible_matches = np.hstack((self._search_for_possible_matches(
                self._preprocess_reduce(to_be_matched)), self._possible_matches))

        if self._verbose:
            tqdm.write('possible matches found   ')
            tqdm.write('')

        if self._possible_matches is None:
            return pd.Series(index=['original_name', 'match_name', 'score', 'match_index'])

        if self._verbose:
            tqdm.write('fuzzy matching...')
            tqdm.write('')
            data_matches = to_be_matched.progress_apply(lambda x: self.fuzzy_matches(
                self._possible_matches[to_be_matched.index.get_loc(x.name), :], x), axis=1)
        else:
            data_matches = to_be_matched.apply(lambda x: self.fuzzy_matches(
                self._possible_matches[to_be_matched.index.get_loc(x.name), :], x), axis=1)

        if self._number_of_matches == 1:
            data_matches = data_matches.rename(columns={
                                               'match_name_0': 'match_name', 'score_0': 'score', 'match_index_0': 'match_index'})

        if self._verbose:
            tqdm.write('done')

        return data_matches

    def fuzzy_matches(self,
                      possible_matches: np.array,
                      to_be_matched: pd.Series) -> pd.Series:
        """
        A method which performs the fuzzy matching between the data in the to_be_matched series as well as
        the indicated indexes of the matching_data points which are possible matching candidates.

        Parameters
        ----------
        possible_matches : np.array
            An array containing the indexes of the matching data with potential matches
        to_be_matched : pd.Series
            The data which should be matched

        Returns
        -------
        pd.Series
            A series containg the match index from the matching_data dataframe. the name in the to_be_matched data,
            the name to which the datapoint was matched and a score between 0 (no match) and 100(perfect match) to 
            indicate the quality of the matches

        """
        
        if len(possible_matches.shape)>1:
            possible_matches = possible_matches[0]
            
        indexes = np.array([[f'match_name_{num}', f'score_{num}', f'match_index_{num}']
                            for num in range(self._number_of_matches)]).flatten()
        match = pd.Series(index=np.append('original_name', indexes))
        match['original_name'] = to_be_matched[self._column_matching]
        list_possible_matches = self._df_matching_data.iloc[possible_matches.flatten(
        ), :][self._column].values

        match_score = self._score_matches(
            to_be_matched[self._column_matching], list_possible_matches)
        ind = self._rate_matches(match_score)

        if len(ind) > 0:
            for num, col_num in enumerate(ind):
                match[f'match_name_{num}'] = list_possible_matches[col_num]
                match[f'score_{num}'] = 100 * np.mean(match_score[col_num, :])
                match[f'match_index_{num}'] = possible_matches[col_num]

        if self._postprocess_common_words or self._postprocess_company_legal_id:
            match = self.postprocess(match)

        return match

    def _score_matches(self,
                       to_be_matched_instance: str,
                       possible_matches: list) -> np.array:
        """
        A method to score a name to_be_matched_instance to a list of possible matches. The scoring is done
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
                      match_score: np.array) -> list:
        """
        Converts the match scores from the score_matches method to a list of indexes of the best scoring 
        matches limited to the _number_of_matches.

        Parameters
        ----------
        match_score : np.array
            An array containing the scores of each of the possible alternatives for each
            of the differnet methods used

        Returns
        -------
        list
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
            ind = np.argmax(match_score, axis=0)
        else:
            ind = np.argsort(np.mean(match_score, axis=1)
                             )[-self._number_of_matches:][::-1]

        return ind

    def postprocess(self,
                    match: pd.Series) -> pd.Series:
        """
        Postprocesses the scores to exclude certain specific company words or the most common words. In 
        this method only the scores are adjusted, the matches still stand.

        Parameters
        ----------
        match : pd.Series
            The series with the possible matches and original scores

        Returns
        -------
        pd.Series
            A new version of the input series with updated scores
        """

        alt_names = []
        for num in range(self._number_of_matches):
            alt_names.append(str(match[f'match_name_{num}']))
        org_name = str(match['original_name'])

        for word in self._word_set:
            org_name = ' '.join(re.sub(r'\b{}\b'.format(
                re.escape(word)), '', org_name).split())
            for num in range(self._number_of_matches):
                alt_names[num] = ' '.join(re.sub(r'\b{}\b'.format(
                    re.escape(word)), '', alt_names[num]).split())

        match_score = self._score_matches(org_name, alt_names)

        for num in range(len(alt_names)):
            match[f'score_{num}'] = 100*np.mean(match_score[num, :])

        return match

    def _vectorise_data(self,
                        transform=True):
        """
        Initialises the TfidfVectorizer, which generates ngrams and weights them based on the occurance.
        Subsequently the matching data will be used to fit the vectoriser and the matching data might also be send
        to the transform_data funtion depending on the transform boolean.

        Parameters
        ----------
        transform : bool 
            A boolean indicating wether or not the data should be transformed after the vectoriser is initialised 
            default: True

        """
        self._vec.fit(self._df_matching_data[self._column].values.flatten())
        if transform:
            self.transform_data()

    def transform_data(self):
        """
        A method which transforms the matching data based on the ngrams transformer. After the 
        transformation (the generation of the ngrams), the data is normalised by dividing each row
        by the sum of the row. Subsequently the data is changed to a coo sparse matrix format with
        the column indices in ascending order.

        """
        ngrams = self._vec.transform(
            self._df_matching_data[self._column].astype(str))
        for i, j in zip(ngrams.indptr[:-1], ngrams.indptr[1:]):
            ngrams.data[i:j] = ngrams.data[i:j]/np.sum(ngrams.data[i:j])
        self._n_grams_matching = ngrams.tocsc()
        if self._low_memory == 0:
            self._n_grams_matching = self._n_grams_matching.tocoo()

    def _search_for_possible_matches(self,
                                     to_be_matched: pd.DataFrame) -> np.array:
        """
        Generates ngrams from the data which should be matched, calculate the cosine simularity
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
                """First the data needs to be transformed to be able to use the sparse cosine simularity. To 
                transform the data, run transform_data or run load_and_process_master_data with transform=True""")

        if self._low_memory == 0:
            results = np.zeros((len(to_be_matched), self._top_n))
            input_data = to_be_matched[self._column_matching]
            for idx, row_name in enumerate(tqdm(input_data, disable=not self._verbose)):
                match_ngrams = self._vec.transform([row_name])
                results[idx, :] = sparse_cosine_top_n(self._n_grams_matching, match_ngrams, self._top_n, self._low_memory, self._verbose)
        else:
            match_ngrams = self._vec.transform(
                to_be_matched[self._column_matching].tolist()).tocsc()
            results = sparse_cosine_top_n(self._n_grams_matching, match_ngrams, self._top_n, self._low_memory, self._verbose)

        return results

    def preprocess(self,
                   df: pd.DataFrame,
                   column_name: str) -> pd.DataFrame:
        """
        Preprocess a dataframe before applying a name matching algorithm. The preprocessing consists of 
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

        df[column_name] = df[column_name].astype(str)
        if self._preprocess_lowercase:
            df[column_name] = df[column_name].str.lower()
        if self._preprocess_punctuations:
            df[column_name] = df[column_name].str.replace('[^\w\s]', '')
            df[column_name] = df[column_name].str.replace(
                '  ', ' ').str.strip()
        if self._preprocess_ascii:
            df[column_name] = df[column_name].apply(lambda string: unicodedata.normalize(
                'NFKD', str(string)).encode('ASCII', 'ignore').decode())

        return df

    def _make_no_scoring_words(self,
                               indicator: str,
                               word_set: set,
                               cut_off=0.2) -> set:
        """
        A method to make a set of words which are not taken into account when scoring matches.

        Parameters
        -------
        indicator: str
            indicator for which types of words should be excluded can be legal for
            legal suffixes or common for the most common words
        word_set: str
            the current word list which should be extended with additional words  
        cut_off: float
            the cut_off percentage of the occurence of the most occuring word for which words are still included 
            in the no_soring_words set
            default=0.01

        Returns
        -------
        Set
            The set of no scoring words

        """

        if indicator == 'legal':
            if self._preprocess_punctuations:
                terms_type = [re.sub(r'[^\w\s]', '', s) for s in functools.reduce(
                    operator.iconcat, terms_by_type.values(), [])]
                terms_country = [re.sub(r'[^\w\s]', '', s) for s in functools.reduce(
                    operator.iconcat, terms_by_country.values(), [])]
            else:
                terms_type = [s for s in functools.reduce(
                    operator.iconcat, terms_by_type.values(), [])]
                terms_country = [s for s in functools.reduce(
                    operator.iconcat, terms_by_country.values(), [])]
            word_set = word_set.union(set(terms_country + terms_type))

        if indicator == 'common':
            word_counts = self._df_matching_data[self._column].str.split(
                expand=True).stack().value_counts()
            word_set = word_set.union(
                set(word_counts[word_counts > np.max(word_counts)*cut_off].index))

        return word_set

