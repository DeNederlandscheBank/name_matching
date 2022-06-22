import numpy as np
import pytest
from scipy.sparse import csc_matrix

from name_matching.sparse_cosine import _sparse_cosine_top_n_standard, _sparse_cosine_low_memory, sparse_cosine_top_n

def assert_values_in_array(A1, A2):
    assert len(A1) == len(A2) 
    A1.sort()
    A2.sort()
    np.testing.assert_array_almost_equal(A1, A2, decimal=2)

@pytest.fixture
def mat_a():
    return csc_matrix(np.array([[0. , 0. , 0. , 0. , 0. , 0.3, 0.2, 0.1, 0.3, 0.4],
                            [0. , 0. , 0. , 0. , 0.6, 0.5, 0. , 0. , 0. , 0. ],
                            [0. , 0. , 0.6, 0.1, 0. , 0.9, 0. , 0. , 0.5, 0. ],
                            [0. , 0.3, 0. , 0.4, 0. , 0.6, 0. , 0.1, 0. , 0. ],
                            [0. , 0. , 0. , 0. , 0. , 0. , 0.3, 0. , 0. , 0. ],
                            [0. , 0. , 0. , 0. , 0. , 0.9, 0. , 0. , 0.2, 0. ],
                            [0.7, 0. , 0. , 0.2, 0.3, 0.9, 0. , 0.3, 0. , 0.5],
                            [0.9, 0.9, 0. , 0.3, 0.9, 0.9, 0. , 0. , 0. , 0. ],
                            [0.9, 0.5, 0. , 0. , 0. , 0.5, 0.4, 0. , 0. , 0.7],
                            [0.1, 0. , 0. , 0.2, 0. , 0.4, 0. , 0.9, 0. , 0.7]]))
@pytest.fixture
def mat_b():
    return csc_matrix(np.array([[0. , 0. , 0.4, 0. , 0.2, 0. , 0. , 0.4, 0. , 0. ],
                                [0. , 0. , 0. , 0.3, 0. , 0. , 0.4, 0. , 0. , 0. ],
                                [0. , 0.9, 0.9, 0.9, 0. , 0.1, 0.2, 0.6, 0. , 0. ],
                                [0. , 0.4, 0. , 0. , 0. , 0. , 0.9, 0. , 0. , 0. ],
                                [0. , 0. , 0. , 0. , 0. , 0. , 0.1, 0. , 0.4, 0. ],
                                [0. , 0. , 0. , 0. , 1. , 0.6, 0.6, 0. , 0. , 0. ],
                                [0. , 0. , 0. , 0.3, 0.6, 0. , 0.9, 0. , 0. , 0. ],
                                [0. , 0. , 0. , 0.9, 0. , 0. , 0. , 0. , 0. , 0.9],
                                [0. , 0. , 0.8, 0. , 0. , 0. , 1. , 0. , 0. , 0. ],
                                [0. , 0. , 0.4, 0. , 0. , 0. , 0.8, 0.3, 0. , 0. ]]))
@pytest.fixture
def result_a_b():
    return np.array([[9., 3., 0., 7., 6., 1., 2., 0., 0., 0.],
                    [8., 4., 0., 9., 7., 6., 3., 2., 0., 0.],
                    [4., 5., 1., 0., 9., 6., 2., 8., 7., 3.],
                    [4., 0., 8., 7., 3., 0., 0., 0., 0., 0.],
                    [5., 2., 8., 4., 0., 0., 0., 0., 0., 0.],
                    [4., 9., 8., 5., 3., 2., 0., 7., 6., 1.],
                    [8., 4., 0., 1., 9., 7., 6., 3., 2., 0.],
                    [8., 0., 9., 7., 6., 3., 2., 0., 0., 0.],
                    [8., 4., 0., 2., 0., 0., 0., 0., 0., 0.],
                    [9., 6., 3., 8., 4., 0., 2., 0., 0., 0.]])
@pytest.fixture
def result_a_b1():
    return np.array([[9.],
                    [8.],
                    [7.],
                    [8.],
                    [2.],
                    [7.],
                    [7.],
                    [9.],
                    [2.],
                    [8.]])
@pytest.fixture
def result_a_b3():
    return np.array([[7., 2., 9.],
                    [4., 3., 8.],
                    [3., 7., 9.],
                    [4., 7., 8.],
                    [5., 2., 0.],
                    [6., 1., 7.],
                    [1., 8., 7.],
                    [6., 8., 9.],
                    [4., 8., 2.],
                    [4., 9., 8.]])
@pytest.fixture
def mat_c():
    return csc_matrix(np.array([[0.2, 0.5, 0.2, 0.1, 0.5, 0. ],
                                [0.2, 0.9, 0.3, 0.4, 0.4, 0.7],
                                [0. , 0. , 0.4, 0. , 0. , 0. ],
                                [0. , 0.5, 0. , 0.3, 0.8, 0. ],
                                [0.7, 0.9, 0. , 0.7, 0.9, 0.2],
                                [0.2, 0.1, 0.8, 0. , 0. , 0.1]]))
@pytest.fixture
def mat_d():
    return csc_matrix(np.array([[0.8, 0. , 0. , 0. , 0.1, 0. ],
                                [0. , 0. , 0. , 0.4, 0. , 0. ],
                                [0.3, 0.4, 0. , 0. , 0. , 0.7],
                                [0. , 0. , 0. , 0. , 0. , 0. ],
                                [0.1, 0.1, 0.4, 0.4, 0. , 0. ],
                                [0.8, 0. , 0.5, 0.8, 0.2, 0. ]]))
@pytest.fixture
def result_c_d():
    return np.array([[3., 5., 4., 1., 0., 0.],
                    [4., 3., 1., 0., 0., 0.],
                    [3., 5., 4., 1., 0., 0.],
                    [0., 0., 0., 0., 0., 0.],
                    [2., 3., 5., 4., 1., 0.],
                    [3., 2., 5., 4., 1., 0.]])
                    
@pytest.fixture
def result_c_d1():
    return np.array([[4],
                    [4],
                    [1],
                    [0],
                    [4],
                    [4]])
                    
@pytest.fixture
def result_c_d4():
    return np.array([[5., 4., 1., 0.],
                    [4., 3., 1., 0.],
                    [3., 4., 1., 0.],
                    [0., 0., 0., 0.],
                    [0., 4., 1., 5.],
                    [0., 4., 1., 5.]])


@pytest.mark.parametrize("top_n, num_rows", 
                          [(10, 10),
                          (10, 8),
                          (10, 7),
                          (10, 1)]
                        )
def test_cosine_standard(top_n, num_rows, mat_a, mat_b, result_a_b):
    np.testing.assert_array_equal(_sparse_cosine_top_n_standard(mat_a, mat_b, num_rows, top_n, False), result_a_b)
    
@pytest.mark.parametrize("top_n, num_rows", 
                          [(1, 10),
                          (1, 8),
                          (1, 7),
                          (1, 1)]
                        )
def test_cosine_standard1(top_n, num_rows, mat_a, mat_b, result_a_b1):
    np.testing.assert_array_equal(_sparse_cosine_top_n_standard(mat_a, mat_b, num_rows, top_n, False), result_a_b1)

@pytest.mark.parametrize("top_n, num_rows", 
                          [(3, 10),
                          (3, 8),
                          (3, 7),
                          (3, 1)]
                        )
def test_cosine_standard3(top_n, num_rows, mat_a, mat_b, result_a_b3):
    np.testing.assert_array_equal(_sparse_cosine_top_n_standard(mat_a, mat_b, num_rows, top_n, False), result_a_b3)
    
@pytest.mark.parametrize("top_n, num_rows", 
                          [(7, 10),
                          (6, 8),
                          (9, 7),
                          (6, 1)]
                        )
def test_cosine_standard_c(top_n, num_rows, mat_c, mat_d, result_c_d):
    np.testing.assert_array_equal(_sparse_cosine_top_n_standard(mat_c, mat_d, num_rows, top_n, False)[:,:6], result_c_d)
    
@pytest.mark.parametrize("top_n, num_rows", 
                          [(4, 5),
                          (4, 4),
                          (4, 3),
                          (4, 1)]
                        )
def test_cosine_standard_c4(top_n, num_rows, mat_c, mat_d, result_c_d4):
    np.testing.assert_array_equal(_sparse_cosine_top_n_standard(mat_c, mat_d, num_rows, top_n, False), result_c_d4)
    
@pytest.mark.parametrize("top_n, num_rows", 
                          [(1, 10),
                          (1, 3),
                          (1, 2),
                          (1, 1)]
                        )
def test_cosine_standard_c1(top_n, num_rows, mat_c, mat_d, result_c_d1):
    np.testing.assert_array_equal(_sparse_cosine_top_n_standard(mat_c, mat_d, num_rows, top_n, False), result_c_d1)

@pytest.mark.parametrize("row", 
                          [[1],[2],[3],[4],[5],[0]]
                        )
def test_cosine_top_n_cd_low_memory(row, mat_a, mat_b):
    mat_a_co = csc_matrix(mat_a).tocoo()
    low_memory_result = _sparse_cosine_low_memory(matrix_row = mat_a_co.row, matrix_col = mat_a_co.col, 
        matrix_data = mat_a_co.data, matrix_len = mat_a_co.shape[0], vector_ind = mat_b[row,:].tocsr().indices, 
        vector_data = mat_b[row,:].tocsr().data)
    ordinary_result = (mat_a * (mat_b).T).todense()[:,row]
    np.testing.assert_array_almost_equal(low_memory_result.reshape(-1,1), ordinary_result, decimal=3)

@pytest.mark.parametrize("top_n, num_rows, row", 
                          [(1, 10, 2),
                          (2, 3, 3),
                          (3, 2, 1),
                          (3, 0, 5),
                          (3, 3, 0),
                          (6, 2, 1),
                          (3, 0, 4),
                          (5, 0, 2),
                          (8, 1, 2)]
                        )
def test_cosine_top_n_cd(top_n, num_rows, row, mat_c, mat_d):
    if num_rows == 0:
        assert_values_in_array(sparse_cosine_top_n(mat_c.tocoo(), mat_d[row,:].tocsr(), top_n, True, num_rows, False).reshape(1,-1), 
        _sparse_cosine_top_n_standard(mat_c, mat_d[row,:], num_rows + 1, top_n, False))
    else:
        np.testing.assert_array_equal(sparse_cosine_top_n(mat_c, mat_d, top_n, False, num_rows, False), _sparse_cosine_top_n_standard(mat_c, mat_d, num_rows, top_n, False))

    
@pytest.mark.parametrize("top_n, num_rows, row", 
                          [(1, 10, 2),
                          (2, 3, 3),
                          (6, 2, 1),
                          (3, 0, 5),
                          (3, 3, 0),
                          (6, 2, 1),
                          (4, 0, 4),
                          (1, 0, 8),
                          (2, 0, 6),
                          (6, 0, 2),
                          (8, 1, 2)]
                        )
def test_cosine_top_n_ab(top_n, num_rows, row, mat_a, mat_b):
    if num_rows == 0:
        assert_values_in_array(sparse_cosine_top_n(mat_a.tocoo(), mat_b[row,:].tocsr(), top_n, True, num_rows, False).reshape(1,-1), 
        _sparse_cosine_top_n_standard(mat_a, mat_b[row,:], num_rows + 1, top_n, False))
    else:
        np.testing.assert_array_equal(sparse_cosine_top_n(mat_a, mat_b, top_n, False, num_rows, False), _sparse_cosine_top_n_standard(mat_a, mat_b, num_rows, top_n, False))
        