import numpy as np
import pandas as pd


def stack_structured_arrays(arrays, append_fields=None):
    """
    Stack numpy structured arrays.

    If `append_fields` is not None, it should contain tuples of (str, array), where str
    is the name of the field that will be added to the stacked structured array, and arr
    is an array that contains a SINGLE element for each structured array in `arrays` (each
    arr must have the same length as `arrays`).

    Parameters
    ----------
    arrays : seq of structured ndarray
        Sequence of structured arrays to be stacked.
    append_fields : None or sequence of (str, arr)
        Append fields to stacked array. The length of each arr in `append_fields` must
        contain an element for each array in `arrays`. The SINGLE ELEMENT will be repeated
        for all rows of the array in `arrays` that it applies to.

    Returns
    -------
    stacked_array : structured ndarray
        Single structured array resulting from stacking all structured arrays.
    """
    n_records = np.array([len(arr) for arr in arrays])
    newdtype = _concat_dtypes(arrays)

    if append_fields is not None:
        for name, arr in append_fields:
            assert len(arr) == len(arrays)
            dtype = (name,) + arr.dtype.descr[0][1:]
            newdtype.append(dtype)

    i = 0
    newarr = np.zeros((n_records.sum(),), dtype=newdtype)
    for arr in arrays:
        newarr[i: i + len(arr)] = arr
        i += len(arr)

    for name, arr in append_fields:
        newarr[name] = np.repeat(arr, n_records)

    return newarr


def _concat_dtypes(arrays):
    ndtypes = np.array([arr.dtype.descr for arr in arrays])
    newdtype = []

    for j in range(ndtypes.shape[1]):
        cols = pd.DataFrame(list(ndtypes[:, j]))

        if len(np.unique(cols.iloc[:, 0])) != 1:
            quit('arrays must have same column names')
        name = np.unique(cols.iloc[:, 0])[0]
        fmt = cols.iloc[:, 1].max()  # autoconvert to largest format
        dtype = (name, fmt)

        if cols.shape[1] == 3:  # shape
            if len(cols.iloc[:, 2].unique()) != 1:
                quit('arrays must have same shapes')
            dtype += (cols.iloc[:, 2].unique()[0],)
        newdtype.append(dtype)

    return newdtype


