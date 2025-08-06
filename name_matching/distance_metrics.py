import distances as nm_dist
from collections import defaultdict


def make_distance_metrics(
    indel:bool|dict=False,
    discounted_levenshtein:bool|dict=False,
    levenshtein:bool|dict=False,
    jaro_winkler:bool|dict=False,
    tichy:bool|dict=False,
    cormodel_z:bool|dict=False,
    iterative_sub_string:bool|dict=False,
    baulieu_xiii:bool|dict=False,
    clement:bool|dict=False,
    dice_asymmetrici:bool|dict=False,
    kuhns_iii:bool|dict=False,
    overlap:bool|dict=False,
    pearson_ii:bool|dict=False,
    weighted_jaccard:bool|dict=False,
    warrens_iv:bool|dict=False,
    bag:bool|dict=False,
    rouge_l:bool|dict=False,
    token_distance:bool|dict=False,
    ratcliff_obershelp:bool|dict=False,
    ncd_bz2:bool|dict=False,
    fuzzy_wuzzy_partial_string:bool|dict=False,
    fuzzy_wuzzy_token_sort:bool|dict=False,
    fuzzy_wuzzy_token_set:bool|dict=False,
    editex:bool|dict=False,
    typo:bool|dict=False,
    lig_3:bool|dict=False,
    ssk:bool|dict=False,
    refined_soundex:bool|dict=False,
    double_metaphone:bool|dict=False,
) -> dict:
    r"""
    A function which returns a dict containing the distance metrics that should be
    used during the fuzzy string matching

    Levenshtein edit distance
        - Indel
        - Discounted Levenshtein
        - levenshtein
        - LIG3
        - Jaro-Winkler
    Block edit distances
        - Tichy
        - CormodeLZ
    Multi-set token-based distance
        - BaulieuXIII
        - Clement
        - DiceAsymmetricI
        - KuhnsIII
        - Overlap
        - PearsonII
        - WeightedJaccard
        - WarrensIV
        - Bag
        - RougeL
        - Token distance
    Subsequence distances
        - IterativeSubString
        - RatcliffObershelp
        - SSK
    Normalized compression distance
        - NCDbz2
    FuzzyWuzzy distances
        - FuzzyWuzzyPartialString
        - FuzzyWuzzyTokenSort
        - FuzzyWuzzyTokenSet
    Ponetic distances
        - RefinedSoundex
        - DoubleMetaphone
    Edit distances
        - Editex
        - Typo


    Parameters
    ----------
    indel: bool
        Boolean indicating whether the Indel method should be used during the
        fuzzy name matching. The indel method is equal to a regular levenshtein
        distance with a twice as high substitution weight. If a dictionary is provided,
        it is used as parameters for the indel distance metric.
        default=False
    discounted_levenshtein: bool
        Boolean indicating whether the DiscountedLevenshtein method should be used
        during the fuzzy name matching. Equal to the regular levenshtein distance,
        only errors later in the string are counted at a discounted rate. To limit 
        the importance of for instance suffix differences. If a dictionary is provided,
        it is used as parameters for the discounted_levenshtein distance metric.
        default=False
    levenshtein: bool
        Boolean indicating whether the Levenshtein method should be used
        during the fuzzy name matching. If a dictionary is provided,
        it is used as parameters for the levenshtein distance metric.
        default=False
    jaro_winkler: bool
        Boolean indicating whether the JaroWinkler method should be used
        during the fuzzy name matching. If a dictionary is provided,
        it is used as parameters for the jaro winkler distance metric.
        default=False
    tichy: bool
        Boolean indicating whether the Tichy method should be used during the
        fuzzy name matching. This algorithm provides a shortest edit distance based
        on substring and add operations. If a dictionary is provided,
        it is used as parameters for the tichy distance metric.
        default=False
    cormodel_z: bool
        Boolean indicating whether the CormodeLZ method should be used during the
        fuzzy name matching. The CormodeLZ distance between strings x and y, is the
        minimum number of single characters or substrings of y or of the partially
        built string which are required to produce x from left to right. If a dictionary is provided,
        it is used as parameters for the cormodel_z distance metric.
        default=False
    iterative_sub_string: bool
        Boolean indicating whether the IterativeSubString method should be used
        during the fuzzy name matching. A method that counts the similarities
        between two strings substrings and subtracts the differences taking into
        account the winkler similarity between the string and the substring. If a dictionary is provided,
        it is used as parameters for the iterative_sub_string distance metric.
        default=False
    baulieu_xiii: bool
        Boolean indicating whether the BaulieuXIII method should be used during
        the fuzzy name matching. The Baulieu XIII distance between two strings is
        given by the following formula: (|X \ Y| + |Y \ X|) / (
        |X ∩ Y| + |X \ Y| + |Y \ X| + |X ∩ Y| ∙ (|X ∩ Y| - 4)^2). If a dictionary is provided,
        it is used as parameters for the baulieu_xiii distance metric.
        default=False
    clement: bool
        Boolean indicating whether the Clement method should be used during the
        fuzzy name matching. The Clement distance between two strings is given
        by the following formula: (|X ∩ Y|/|X|)*(1-|X|/|N|) + (|(N \ X) \ Y|/|N \ X|) *
        (1-|N \ X|/|N|). If a dictionary is provided,
        it is used as parameters for the clement distance metric.
        default=False
    dice_asymmetrici: bool
        Boolean indicating whether the DiceAsymmetricI method should be used during
        the fuzzy name matching. The Dice asymmetric similarity is given be |X ∩ Y|/|X|. If a dictionary is provided,
        it is used as parameters for the dice_asymmetrici distance metric.
        default=False
    kuhns_iii: bool
        Boolean indicating whether the KuhnsIII method should be used during the
        fuzzy name matching. If a dictionary is provided,
        it is used as parameters for the kuhns_iii distance metric.
        default=False
    overlap: bool
        Boolean indicating whether the Overlap method should be used during the
        fuzzy name matching. The overlap distance is given by: |X ∩ Y|/min(|X|,|Y|). If a dictionary is provided,
        it is used as parameters for the overlap distance metric.
        default=True
    pearson_ii: bool
        Boolean indicating whether the PearsonII method should be used during the
        fuzzy name matching. This algorithm is based on the Phi coefficient or the
        mean square contingency. If a dictionary is provided,
        it is used as parameters for the pearson_ii distance metric.
        default=False
    weighted_jaccard: bool
        Boolean indicating whether the WeightedJaccard method should be used during
        the fuzzy name matching. This is the Jaccard distance only using a wheighing
        for the differences of 3. If a dictionary is provided,
        it is used as parameters for the weighted_jaccard distance metric.
        default=True
    warrens_iv: bool
        Boolean indicating whether the WarrensIV method should be used during the
        fuzzy name matching. If a dictionary is provided,
        it is used as parameters for the warrens_iv distance metric.
        default=False
    bag: bool
        Boolean indicating whether the Bag method should be used during the fuzzy
            name matching. Is a simplification of the regular edit distance by using
            a similarity tree structure. If a dictionary is provided,
        it is used as parameters for the bag distance metric.
        default=False
    rouge_l: bool
        Boolean indicating whether the ROUGE-L method should be used during the
        fuzzy name matching. The ROGUE-L method is a measure that counts the longest
        substring between to strings. If a dictionary is provided,
        it is used as parameters for the rouge_l distance metric.
        default=False
    token_distance: bool
        Boolean indicating whether the a token distance method should be used during the
        fuzzy name matching. If a dictionary is provided, it is used as parameters for
        the tokenised distance metrics.
        default=False
    ratcliff_obershelp: bool
        Boolean indicating whether the RatcliffObershelp method should be used
        during the fuzzy name matching. This method finds the longest common substring
        and evaluates the longest common substrings to the right and the left of the
        original longest common substring. If a dictionary is provided,
        it is used as parameters for the ratcliff_obershelp distance metric.
        default=True
    ncd_bz2: bool
        Boolean indicating whether the NCDbz2 method should be used during the
        fuzzy name matching. Applies the Burrows-Wheeler transform to the strings and
        subsequently returns the normalised compression distance. If a dictionary is provided,
        it is used as parameters for the ncd_bz2 distance metric.
        default=False
    fuzzy_wuzzy_partial_string: bool
        Boolean indicating whether the FuzzyWuzzyPartialString method should be used
        during the fuzzy name matching. This methods takes the length of the longest
        common substring and divides it over the minimum of the length of each of
        the two strings. If a dictionary is provided,
        it is used as parameters for the fuzzy_wuzzy_partial_string distance metric.
        default=False
    fuzzy_wuzzy_token_sort: bool
        Boolean indicating whether the FuzzyWuzzyTokenSort method should be used
        during the fuzzy name matching. This tokenizes the words in the string
        and sorts them, subsequently a hamming distance is calculated. If a dictionary is provided,
        it is used as parameters for the fuzzy_wuzzy_token_sort distance metric.
        default=True
    fuzzy_wuzzy_token_set: bool
        Boolean indicating whether the FuzzyWuzzyTokenSet method should be used
        during the fuzzy name matching. This method tokenizes the strings and
        find the largest intersection of the two substrings and divides it over
        the length of the shortest string. If a dictionary is provided,
        it is used as parameters for the fuzzy_wuzzy_token_set distance metric.
        default=False
    editex: bool
        Boolean indicating whether the Editex method should be used during the
        fuzzy name matching. If a dictionary is provided,
        it is used as parameters for the editex distance metric.
        default=True
    typo: bool
        Boolean indicating whether the Typo method should be used during the
        fuzzy name matching. The typo distance is calculated based on the distance
        on a keyboard between edits. If a dictionary is provided,
        it is used as parameters for the typo distance metric.
        default=False
    lig_3: bool
        Boolean indicating whether the LIG3 method should be used during the fuzzy
        name matching. If a dictionary is provided,
        it is used as parameters for the lig_3 distance metric.
        default=False
    ssk: bool
        Boolean indicating whether the SSK method should be used during the fuzzy
        name matching. The ssk algorithm looks at the string kernel generated by all
        the possible different subsequences present between the two strings. If a dictionary is provided,
        it is used as parameters for the ssk distance metric.
        default=False
    refined_soundex: bool
        Boolean indicating whether the string should be represented by the RefinedSoundex
        phonetix translation and the Levensthein distance of the translated strings should
        be included in the fuzzy matching process. If a dictionary is provided,
        it is used as parameters for the refined_soundex distance metric.
        default=False
    double_metaphone: bool
        Boolean indicating whether the string should be represented by the DoubleMetaphone
        phonetix translation and the Levensthein distance of the translated strings should
        be included in the fuzzy matching process. If a dictionary is provided,
        it is used as parameters for the Double Metaphone distance metric.
        default=False

    """
    distance_metrics = defaultdict(list)
    if indel:
        if isinstance(indel, dict):
            distance_metrics["Levenshtein"].append(nm_dist.Indel(**indel))
        else:
            distance_metrics["Levenshtein"].append(nm_dist.Indel())
    if discounted_levenshtein:
        if isinstance(discounted_levenshtein, dict):
            distance_metrics["Levenshtein"].append(nm_dist.DiscountedLevenshtein(**discounted_levenshtein))
        else:
            distance_metrics["Levenshtein"].append(nm_dist.DiscountedLevenshtein())
    if levenshtein:
        if isinstance(levenshtein, dict):
            distance_metrics["Levenshtein"].append(nm_dist.Levenshtein(**levenshtein))
        else:
            distance_metrics["Levenshtein"].append(nm_dist.Levenshtein())
    if jaro_winkler:
        if isinstance(jaro_winkler, dict):
            distance_metrics["Levenshtein"].append(nm_dist.JaroWinkler(**jaro_winkler))
        else:
            distance_metrics["Levenshtein"].append(nm_dist.JaroWinkler())
    if cormodel_z:
        if isinstance(cormodel_z, dict):
            distance_metrics["block"].append(nm_dist.CormodeLZ(**cormodel_z))
        else:
            distance_metrics["block"].append(nm_dist.CormodeLZ())
    if tichy:
        if isinstance(tichy, dict):
            distance_metrics["block"].append(nm_dist.Tichy(**tichy))
        else:
            distance_metrics["block"].append(nm_dist.Tichy())
    if iterative_sub_string:
        if isinstance(iterative_sub_string, dict):
            distance_metrics["Subsequence"].append(nm_dist.IterativeSubString(**iterative_sub_string))
        else:
            distance_metrics["Subsequence"].append(nm_dist.IterativeSubString())
    if baulieu_xiii:
        if isinstance(baulieu_xiii, dict):
            distance_metrics["multiset"].append(nm_dist.BaulieuXIII(**baulieu_xiii))
        else:
            distance_metrics["multiset"].append(nm_dist.BaulieuXIII())
    if clement:
        if isinstance(clement, dict):
            distance_metrics["multiset"].append(nm_dist.Clement(**clement))
        else:
            distance_metrics["multiset"].append(nm_dist.Clement())
    if dice_asymmetrici:
        if isinstance(dice_asymmetrici, dict):
            distance_metrics["multiset"].append(nm_dist.DiceAsymmetricI(**dice_asymmetrici))
        else:
            distance_metrics["multiset"].append(nm_dist.DiceAsymmetricI())
    if kuhns_iii:
        if isinstance(kuhns_iii, dict):
            distance_metrics["multiset"].append(nm_dist.KuhnsIII(**kuhns_iii))
        else:
            distance_metrics["multiset"].append(nm_dist.KuhnsIII())
    if overlap:
        if isinstance(overlap, dict):
            distance_metrics["multiset"].append(nm_dist.Overlap(**overlap))
        else:
            distance_metrics["multiset"].append(nm_dist.Overlap())
    if pearson_ii:
        if isinstance(pearson_ii, dict):
            distance_metrics["multiset"].append(nm_dist.PearsonII(**pearson_ii))
        else:
            distance_metrics["multiset"].append(nm_dist.PearsonII())
    if weighted_jaccard:
        if isinstance(weighted_jaccard, dict):
            distance_metrics["multiset"].append(nm_dist.WeightedJaccard(**weighted_jaccard))
        else:
            distance_metrics["multiset"].append(nm_dist.WeightedJaccard())
    if warrens_iv:
        if isinstance(warrens_iv, dict):
            distance_metrics["multiset"].append(nm_dist.WarrensIV(**warrens_iv))
        else:
            distance_metrics["multiset"].append(nm_dist.WarrensIV())
    if bag:
        if isinstance(bag, dict):
            distance_metrics["multiset"].append(nm_dist.Bag(**bag))
        else:
            distance_metrics["multiset"].append(nm_dist.Bag())
    if rouge_l:
        if isinstance(rouge_l, dict):
            distance_metrics["multiset"].append(nm_dist.RougeL(**rouge_l))
        else:
            distance_metrics["multiset"].append(nm_dist.RougeL())
    if token_distance:
        if isinstance(token_distance, dict):
            distance_metrics["multiset"].append(nm_dist._TokenDistance(**token_distance))
        else:
            distance_metrics["multiset"].append(nm_dist._TokenDistance())
    if ratcliff_obershelp:
        if isinstance(ratcliff_obershelp, dict):
            distance_metrics["Subsequence"].append(nm_dist.RatcliffObershelp(**ratcliff_obershelp))
        else:
            distance_metrics["Subsequence"].append(nm_dist.RatcliffObershelp())
    if ncd_bz2:
        if isinstance(ncd_bz2, dict):
            distance_metrics["compression"].append(nm_dist.NCDbz2(**ncd_bz2))
        else:
            distance_metrics["compression"].append(nm_dist.NCDbz2())
    if fuzzy_wuzzy_partial_string:
        if isinstance(fuzzy_wuzzy_partial_string, dict):
            distance_metrics["fuzzy"].append(nm_dist.FuzzyWuzzyPartialString(**fuzzy_wuzzy_partial_string))
        else:
            distance_metrics["fuzzy"].append(nm_dist.FuzzyWuzzyPartialString())
    if fuzzy_wuzzy_token_sort:
        if isinstance(fuzzy_wuzzy_token_sort, dict):
            distance_metrics["fuzzy"].append(nm_dist.FuzzyWuzzyTokenSort(**fuzzy_wuzzy_token_sort))
        else:
            distance_metrics["fuzzy"].append(nm_dist.FuzzyWuzzyTokenSort())
    if fuzzy_wuzzy_token_set:
        if isinstance(fuzzy_wuzzy_token_set, dict):
            distance_metrics["fuzzy"].append(nm_dist.FuzzyWuzzyTokenSet(**fuzzy_wuzzy_token_set))
        else:
            distance_metrics["fuzzy"].append(nm_dist.FuzzyWuzzyTokenSet())
    if editex:
        if isinstance(editex, dict):
            distance_metrics["edit"].append(nm_dist.Editex(**editex))
        else:
            distance_metrics["edit"].append(nm_dist.Editex())
    if typo:
        if isinstance(typo, dict):
            distance_metrics["edit"].append(nm_dist.Typo(**typo))
        else:
            distance_metrics["edit"].append(nm_dist.Typo())
    if lig_3:
        if isinstance(lig_3, dict):
            distance_metrics["Levenshtein"].append(nm_dist.LIG3(**lig_3))
        else:
            distance_metrics["Levenshtein"].append(nm_dist.LIG3())
    if ssk:
        if isinstance(ssk, dict):
            distance_metrics["Subsequence"].append(nm_dist.SSK(**ssk))
        else:
            distance_metrics["Subsequence"].append(nm_dist.SSK())
    if refined_soundex:
        if not isinstance(refined_soundex, dict):
            refined_soundex = {'max_length':30}
        distance_metrics["phonetic"].append(
            nm_dist.PhoneticDistance(
                transforms=nm_dist.RefinedSoundex(**refined_soundex),
                metric=nm_dist.Levenshtein(),
                encode_alpha=True,
            )
        )
    if double_metaphone:
        if not isinstance(double_metaphone, dict):
            double_metaphone = {'max_length':30}
        distance_metrics["phonetic"].append(
            nm_dist.PhoneticDistance(
                transforms=nm_dist.DoubleMetaphone(**double_metaphone),
                metric=nm_dist.Levenshtein(),
                encode_alpha=True,
            )
        )

    return distance_metrics

