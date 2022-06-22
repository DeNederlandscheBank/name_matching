import numpy as np
from tqdm import tqdm
# from numba import jit
from scipy.sparse import csc_matrix, coo_matrix
from typing import Union

# @jit(nopython=True, fastmath=True)
def _sparse_cosine_low_memory(matrix_row: np.array,
                              matrix_col: np.array,
                              matrix_data: np.array,
                              matrix_len: int,
                              vector_ind: np.array,
                              vector_data: np.array) -> np.array:
    """
    A sparse cosine simularity calculation between a matrix and a vector. The sparse matrix should be sorted
    in ascending order based on the matrix_col values. The vector should be sorted based on the indexes in 
    ascending order.

    Parameters
    ----------
    matrix_row : np.array
        The row indices of the ngrams matrix of the matching data
    matrix_col : np.array
        The column indices of the ngrams matrix of the matching data in ascending order
    matrix_data : np.array
        The data of the ngrams matrix of the matching data
    matrix_len : int
        The length (number of rows) of the ngrams matrix of the matching data
    vector_ind : np.array
        The indices of the ngrams vector of the to be matched data
    vector_data : np.array
        The data of the ngrams vector of the to be matched data

    Returns
    -------
    np.array
        The cosine simularity between each of the rows of the matrix and the vector

    """
    ind = 0
    res = np.zeros(matrix_len, np.float32)
    for mat_ind in range(len(matrix_col)):
        col = matrix_col[mat_ind]
        if col > vector_ind[ind]:
            ind = ind + 1
            if ind == len(vector_ind):
                break
        if col == vector_ind[ind]:
            res[matrix_row[mat_ind]] = res[matrix_row[mat_ind]] + \
                matrix_data[mat_ind] * vector_data[ind]

    return res


def _sparse_cosine_top_n_standard(matrix_a: csc_matrix,
                                  matrix_b: csc_matrix,
                                  number_of_rows_at_once: int,
                                  top_n: int,
                                  verbose: bool) -> np.array:
    """
    A function for sparse matrix multiplication followed by an argpartition to
    only take the top_n indexes.

    Parameters
    -------
    matrix_a: csc_matric
        The largest sparse csc matrix which should be multiplied
    matrix_b: csc_matric
        The smallest sparse csc matrix which should be multiplied
    number_of_rows_at_once: int
        The number of rows which should be processed at once, a lower
        number of rows reduces the memory usage
    top_n: int
        The best n matches that should be returned
    verbose: bool
        A boolean indicating whether the progress should be printed

    Returns
    -------
    np.array
        The indexes for the n best sparse cosine matches between matrix a and b

    """

    results_arg = np.zeros(
        (matrix_b.shape[0], top_n), dtype=np.float32)

    # Split up the matrice in a certain number of rows
    for j in tqdm(range(0, matrix_b.shape[0], number_of_rows_at_once), disable=not verbose):
        number_of_rows_at_once_min = min(
            [number_of_rows_at_once, matrix_b.shape[0]-j])
        matrix_b_temp = matrix_b[j:j+number_of_rows_at_once_min, :]

        # Calculate the matrix dot product
        results_full = (matrix_a * (matrix_b_temp.T)).tocsc()

        # For each of the rows of the original matrix select the argpartition
        for i in range(number_of_rows_at_once_min):
            results_full_temp = results_full.data[results_full.indptr[i]:results_full.indptr[i+1]]

            # If there are more results then top_n only select the top_n results
            if len(results_full_temp) > top_n:
                ind = results_full.indices[results_full.indptr[i]:results_full.indptr[i+1]]
                results_arg[j + i, :] = ind[np.argpartition(
                    results_full_temp, -top_n)[-top_n:]]
            
            # else just select all the results
            else:
                results_arg[j + i, :len(results_full_temp)
                            ] = results_full.indices[results_full.indptr[i]:results_full.indptr[i+1]]
    return results_arg

def sparse_cosine_top_n(matrix_a: Union[csc_matrix, coo_matrix], 
                        matrix_b: csc_matrix, 
                        top_n: int, 
                        low_memory: bool,
                        number_of_rows: int,
                        verbose: bool):
    """
    Calculates the top_n cosine matches between matrix_a and matrix_b. Takes into account
    the amount of  memory that should be used based on the low_memory int

    Parameters
    -------
    matrix_a: csc_matric
        The largest sparse csc matrix which should be multiplied
    matrix_b: csc_matric
        The smallest sparse csc matrix which should be multiplied
    top_n: int
        The best n matches that should be returned
    low_memory: bool
        A bool indicating whether the low memory sparse cosine approach should be used
    number_of_rows: int
        An int inidcating the number of rows which should be 
        processed at once when calculating the cosine simalarity
    verbose: bool
        A boolean indicating whether the progress should be printed

    Returns
    -------
    np.array
        The indexes for the n best sparse cosine matches between matrix a and b

    """
    if low_memory:
        matrix_b.sort_indices()
        res = _sparse_cosine_low_memory(matrix_a.row, matrix_a.col, matrix_a.data,
                        matrix_a.shape[0], matrix_b.indices, matrix_b.data)

        top_n_adjusted = -np.min([top_n, len(res)])

        return np.argpartition(res, top_n_adjusted, axis=0)[top_n_adjusted:]
    else:
        return _sparse_cosine_top_n_standard(matrix_a, matrix_b, number_of_rows, top_n, verbose)