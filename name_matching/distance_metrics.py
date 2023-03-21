from distances import Indel, DiscountedLevenshtein, CormodeLZ, Tichy, IterativeSubString, BaulieuXIII, Clement, DiceAsymmetricI, KuhnsIII, Overlap, PearsonII, WeightedJaccard, WarrensIV, Bag, RougeL, RatcliffObershelp, NCDbz2, FuzzyWuzzyPartialString, FuzzyWuzzyTokenSort, FuzzyWuzzyTokenSet, Editex, Typo,LIG3, SSK, Levenshtein, DoubleMetaphone, RefinedSoundex, PhoneticDistance
from collections import defaultdict

def make_distance_metrics(indel=False,
                          discounted_levenshtein=False,
                          tichy=False,
                          cormodel_z=False,
                          iterative_sub_string=False,
                          baulieu_xiii=False,
                          clement=False,
                          dice_asymmetrici=False,
                          kuhns_iii=False,
                          overlap=False,
                          pearson_ii=False,
                          weighted_jaccard=False,
                          warrens_iv=False,
                          bag=False,
                          rouge_l=False,
                          ratcliff_obershelp=False,
                          ncd_bz2=False,
                          fuzzy_wuzzy_partial_string=False,
                          fuzzy_wuzzy_token_sort=False,
                          fuzzy_wuzzy_token_set=False,
                          editex=False,
                          typo=False,
                          lig_3=False,
                          ssk=False,
                          refined_soundex=False,
                          double_metaphone=False) -> dict:
    r"""
    A function which returns a dict containing the distance metrics that should be 
    used during the fuzzy string matching

    Levenshtein edit distance
        - Indel
        - Discounted Levenshtein
        - LIG3
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
        distance with a twice as high substitution weight 
        default=False
    discounted_levenshtein: bool
        Boolean indicating whether the DiscountedLevenshtein method should be used 
        during the fuzzy name matching. Equal to the regular levenshtein distance,
        only errors later in the string are counted at a discounted rate. To for
        instance limit the importance of suffix differences
        default=False
    tichy: bool
        Boolean indicating whether the Tichy method should be used during the 
        fuzzy name matching. This algorithm provides a shortest edit distance based
        on substring and add operations.
        default=False
    cormodel_z: bool
        Boolean indicating whether the CormodeLZ method should be used during the 
        fuzzy name matching. The CormodeLZ distance between strings x and y, is the 
        minimum number of single characters or substrings of y or of the partially 
        built string which are required to produce x from left to right.
        default=False
    iterative_sub_string: bool
        Boolean indicating whether the IterativeSubString method should be used 
        during the fuzzy name matching. A method that counts the similarities 
        between two strings substrings and subtracts the differences taking into
        account the winkler similarity between the string and the substring.
        default=False
    baulieu_xiii: bool
        Boolean indicating whether the BaulieuXIII method should be used during 
        the fuzzy name matching. The Baulieu XIII distance between two strings is
        given by the following formula: (|X \ Y| + |Y \ X|) / (
        |X ∩ Y| + |X \ Y| + |Y \ X| + |X ∩ Y| ∙ (|X ∩ Y| - 4)^2)
        default=False
    clement: bool
        Boolean indicating whether the Clement method should be used during the
        fuzzy name matching. The Clement distance between two strings is given 
        by the following formula: (|X ∩ Y|/|X|)*(1-|X|/|N|) + (|(N \ X) \ Y|/|N \ X|) *
        (1-|N \ X|/|N|)
        default=False
    dice_asymmetrici: bool
        Boolean indicating whether the DiceAsymmetricI method should be used during 
        the fuzzy name matching. The Dice asymmetric similarity is given be |X ∩ Y|/|X|
        default=False
    kuhns_iii: bool
        Boolean indicating whether the KuhnsIII method should be used during the 
        fuzzy name matching
        default=False
    overlap: bool
        Boolean indicating whether the Overlap method should be used during the 
        fuzzy name matching. The overlap distance is given by: |X ∩ Y|/min(|X|,|Y|)
        default=True
    pearson_ii: bool
        Boolean indicating whether the PearsonII method should be used during the 
        fuzzy name matching. This algorithm is based on the Phi coefficient or the 
        mean square contingency
        default=False
    weighted_jaccard: bool
        Boolean indicating whether the WeightedJaccard method should be used during 
        the fuzzy name matching. This is the Jaccard distance only using a wheighing 
        for the differences of 3.
        default=True
    warrens_iv: bool
        Boolean indicating whether the WarrensIV method should be used during the 
        fuzzy name matching
        default=False
    bag: bool
        Boolean indicating whether the Bag method should be used during the fuzzy
            name matching. Is a simplification of the regular edit distance by using
            a similarity tree structure.
        default=False
    rouge_l: bool
        Boolean indicating whether the ROUGE-L method should be used during the 
        fuzzy name matching. The ROGUE-L method is a measure that counts the longest
        substring between to strings
        default=False
    ratcliff_obershelp: bool
        Boolean indicating whether the RatcliffObershelp method should be used 
        during the fuzzy name matching. This method finds the longest common substring
        and evaluates the longest common substrings to the right and the left of the 
        original longest common substring
        default=True
    ncd_bz2: bool
        Boolean indicating whether the NCDbz2 method should be used during the 
        fuzzy name matching. Applies the Burrows-Wheeler transform to the strings and 
        subsequently returns the normalised compression distance.
        default=False
    fuzzy_wuzzy_partial_string: bool
        Boolean indicating whether the FuzzyWuzzyPartialString method should be used
        during the fuzzy name matching. This methods takes the length of the longest 
        common substring and divides it over the minimum of the length of each of 
        the two strings.
        default=False
    fuzzy_wuzzy_token_sort: bool
        Boolean indicating whether the FuzzyWuzzyTokenSort method should be used 
        during the fuzzy name matching. This tokenizes the words in the string
        and sorts them, subsequently a hamming distance is calculated
        default=True
    fuzzy_wuzzy_token_set: bool
        Boolean indicating whether the FuzzyWuzzyTokenSet method should be used 
        during the fuzzy name matching. This method tokenizes the strings and 
        find the largest intersection of the two substrings and divides it over 
        the length of the shortest string
        default=False
    editex: bool
        Boolean indicating whether the Editex method should be used during the 
        fuzzy name matching
        default=True
    typo: bool
        Boolean indicating whether the Typo method should be used during the 
        fuzzy name matching. The typo distance is calculated based on the distance
        on a keyboard between edits.
        default=False
    lig_3: bool
        Boolean indicating whether the LIG3 method should be used during the fuzzy 
        name matching
        default=False
    ssk: bool
        Boolean indicating whether the SSK method should be used during the fuzzy 
        name matching. The ssk algorithm looks at the string kernel generated by all 
        the possible different subsequences present between the two strings.
        default=False
    refined_soundex: bool
        Boolean indicating whether the string should be represented by the RefinedSoundex
        phonetix translation and the Levensthein distance of the translated strings should
        be included in the fuzzy matching process
        default=False
    double_metaphone: bool
        Boolean indicating whether the string should be represented by the DoubleMetaphone
        phonetix translation and the Levensthein distance of the translated strings should
        be included in the fuzzy matching process
        default=False

    """
    distance_metrics = defaultdict(list)
    if indel:
        distance_metrics['Levenshtein'].append(Indel())
    if discounted_levenshtein:
        distance_metrics['Levenshtein'].append(
            DiscountedLevenshtein())
    if cormodel_z:
        distance_metrics['block'].append(CormodeLZ())
    if tichy:
        distance_metrics['block'].append(Tichy())
    if iterative_sub_string:
        distance_metrics['Subsequence'].append(
            IterativeSubString())
    if baulieu_xiii:
        distance_metrics['multiset'].append(BaulieuXIII())
    if clement:
        distance_metrics['multiset'].append(Clement())
    if dice_asymmetrici:
        distance_metrics['multiset'].append(DiceAsymmetricI())
    if kuhns_iii:
        distance_metrics['multiset'].append(KuhnsIII())
    if overlap:
        distance_metrics['multiset'].append(Overlap())
    if pearson_ii:
        distance_metrics['multiset'].append(PearsonII())
    if weighted_jaccard:
        distance_metrics['multiset'].append(WeightedJaccard())
    if warrens_iv:
        distance_metrics['multiset'].append(WarrensIV())
    if bag:
        distance_metrics['multiset'].append(Bag())
    if rouge_l:
        distance_metrics['multiset'].append(RougeL())
    if ratcliff_obershelp:
        distance_metrics['Subsequence'].append(
            RatcliffObershelp())
    if ncd_bz2:
        distance_metrics['compression'].append(NCDbz2())
    if fuzzy_wuzzy_partial_string:
        distance_metrics['fuzzy'].append(
            FuzzyWuzzyPartialString())
    if fuzzy_wuzzy_token_sort:
        distance_metrics['fuzzy'].append(FuzzyWuzzyTokenSort())
    if fuzzy_wuzzy_token_set:
        distance_metrics['fuzzy'].append(FuzzyWuzzyTokenSet())
    if editex:
        distance_metrics['edit'].append(Editex())
    if typo:
        distance_metrics['edit'].append(Typo())
    if lig_3:
        distance_metrics['Levenshtein'].append(LIG3())
    if ssk:
        distance_metrics['Subsequence'].append(SSK())
    if refined_soundex:
        distance_metrics['phonetic'].append(PhoneticDistance(
            transforms=RefinedSoundex(max_length=30), metric=Levenshtein(), encode_alpha=True))
    if double_metaphone:
        distance_metrics['phonetic'].append(PhoneticDistance(
            transforms=DoubleMetaphone(max_length=30), metric=Levenshtein(), encode_alpha=True))

    return distance_metrics
