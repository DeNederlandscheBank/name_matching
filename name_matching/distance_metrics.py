import abydos.distance as abd
import abydos.phonetic as abp
from collections import defaultdict

def make_distance_metrics(indel=False,
                          discounted_levenshtein=False,
                          tichy=False,
                          cormodeL_z=False,
                          iterative_sub_string=False,
                          baulieu_xiii=False,
                          clement=False,
                          dice_asymmetricI=False,
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
    """
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
        fuzzy name matching
        default=False
    discounted_levenshtein: bool
        Boolean indicating whether the DiscountedLevenshtein method should be used 
        during the fuzzy name matching
        default=False
    tichy: bool
        Boolean indicating whether the Tichy method should be used during the 
        fuzzy name matching
        default=False
    cormodeL_z: bool
        Boolean indicating whether the CormodeLZ method should be used during the 
        fuzzy name matching
        default=False
    iterative_sub_string: bool
        Boolean indicating whether the IterativeSubString method should be used 
        during the fuzzy name matching
        default=False
    baulieu_xiii: bool
        Boolean indicating whether the BaulieuXIII method should be used during 
        the fuzzy name matching
        default=False
    clement: bool
        Boolean indicating whether the Clement method should be used during the
            fuzzy name matching
        default=False
    dice_asymmetricI: bool
        Boolean indicating whether the DiceAsymmetricI method should be used during 
        the fuzzy name matching
        default=False
    kuhns_iii: bool
        Boolean indicating whether the KuhnsIII method should be used during the 
        fuzzy name matching
        default=False
    overlap: bool
        Boolean indicating whether the Overlap method should be used during the 
        fuzzy name matching
        default=True
    pearson_ii: bool
        Boolean indicating whether the PearsonII method should be used during the 
        fuzzy name matching
        default=False
    weighted_jaccard: bool
        Boolean indicating whether the WeightedJaccard method should be used during 
        the fuzzy name matching
        default=True
    warrens_iv: bool
        Boolean indicating whether the WarrensIV method should be used during the 
        fuzzy name matching
        default=False
    bag: bool
        Boolean indicating whether the Bag method should be used during the fuzzy
            name matching
        default=False
    rouge_l: bool
        Boolean indicating whether the RougeL method should be used during the 
        fuzzy name matching
        default=False
    ratcliff_obershelp: bool
        Boolean indicating whether the RatcliffObershelp method should be used 
        during the fuzzy name matching
        default=True
    ncd_bz2: bool
        Boolean indicating whether the NCDbz2 method should be used during the 
        fuzzy name matching
        default=False
    fuzzy_wuzzy_partial_string: bool
        Boolean indicating whether the FuzzyWuzzyPartialString method should be used
            during the fuzzy name matching
        default=False
    fuzzy_wuzzy_token_sort: bool
        Boolean indicating whether the FuzzyWuzzyTokenSort method should be used 
        during the fuzzy name matching
        default=True
    fuzzy_wuzzy_token_set: bool
        Boolean indicating whether the FuzzyWuzzyTokenSet method should be used 
        during the fuzzy name matching
        default=False
    editex: bool
        Boolean indicating whether the Editex method should be used during the 
        fuzzy name matching
        default=True
    typo: bool
        Boolean indicating whether the Typo method should be used during the 
        fuzzy name matching
        default=False
    lig_3: bool
        Boolean indicating whether the LIG3 method should be used during the fuzzy 
        name matching
        default=False
    ssk: bool
        Boolean indicating whether the SSK method should be used during the fuzzy 
        name matching
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
        distance_metrics['Levenshtein'].append(abd.Indel())
    if discounted_levenshtein:
        distance_metrics['Levenshtein'].append(
            abd.DiscountedLevenshtein())
    if cormodeL_z:
        distance_metrics['block'].append(abd.CormodeLZ())
    if tichy:
        distance_metrics['block'].append(abd.Tichy())
    if iterative_sub_string:
        distance_metrics['Subsequence'].append(
            abd.IterativeSubString())
    if baulieu_xiii:
        distance_metrics['multiset'].append(abd.BaulieuXIII())
    if clement:
        distance_metrics['multiset'].append(abd.Clement())
    if dice_asymmetricI:
        distance_metrics['multiset'].append(abd.DiceAsymmetricI())
    if kuhns_iii:
        distance_metrics['multiset'].append(abd.KuhnsIII())
    if overlap:
        distance_metrics['multiset'].append(abd.Overlap())
    if pearson_ii:
        distance_metrics['multiset'].append(abd.PearsonII())
    if weighted_jaccard:
        distance_metrics['multiset'].append(abd.WeightedJaccard())
    if warrens_iv:
        distance_metrics['multiset'].append(abd.WarrensIV())
    if bag:
        distance_metrics['multiset'].append(abd.Bag())
    if rouge_l:
        distance_metrics['multiset'].append(abd.RougeL())
    if ratcliff_obershelp:
        distance_metrics['Subsequence'].append(
            abd.RatcliffObershelp())
    if ncd_bz2:
        distance_metrics['compression'].append(abd.NCDbz2())
    if fuzzy_wuzzy_partial_string:
        distance_metrics['fuzzy'].append(
            abd.FuzzyWuzzyPartialString())
    if fuzzy_wuzzy_token_sort:
        distance_metrics['fuzzy'].append(abd.FuzzyWuzzyTokenSort())
    if fuzzy_wuzzy_token_set:
        distance_metrics['fuzzy'].append(abd.FuzzyWuzzyTokenSet())
    if editex:
        distance_metrics['edit'].append(abd.Editex())
    if typo:
        distance_metrics['edit'].append(abd.Typo())
    if lig_3:
        distance_metrics['Levenshtein'].append(abd.LIG3())
    if ssk:
        distance_metrics['Subsequence'].append(abd.SSK())
    if refined_soundex:
        distance_metrics['phonetic'].append(abd.PhoneticDistance(
            transforms=abp.RefinedSoundex(max_length=30), metric=abd.Levenshtein(), encode_alpha=True))
    if double_metaphone:
        distance_metrics['phonetic'].append(abd.PhoneticDistance(
            transforms=abp.DoubleMetaphone(max_length=30), metric=abd.Levenshtein(), encode_alpha=True))

    return distance_metrics