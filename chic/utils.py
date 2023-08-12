"""
30.07.23
@tcnicholas
Utility functions.
"""


import warnings
from math import log10
from time import perf_counter
from collections import Counter
from string import ascii_lowercase
from typing import List, Callable, Any, Union, TypeVar, Dict

import numpy as np

T = TypeVar('T', bound=Callable[..., Any])


class Colours:
    """
    A bit extra but it makes the Jupyter Notebooks look snazzy.
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    

    @staticmethod
    def colourise(text: str, colour: str) -> str:
        """
        Colours the given text with the provided colour.

        :param text: The text to colourise.
        :param colour: The colour to use.
        :return: The coloured text.
        """
        return f"{colour}{text}{Colours.ENDC}"
    

class CoarseGrainingMethodRegistry:

    def __init__(self):
        self.methods = {}


    def register(self, name, bead_type):
        def inner(func):
            self.methods[name] = {"func": func, "bead_type": bead_type}
            return func
        return inner
    

    def register_from_file(self, name, func, bead_type):
        self.methods[name] = {"func": func, "bead_type": bead_type}
        return func


    def get_method(self, name):
        return self.methods.get(name, {}).get('func')


    def get_bead_type(self, name):
        return self.methods.get(name, {}).get('bead_type')
    

    def available_methods(self):
        methods = list(self.methods.keys())
        bead_types = [self.get_bead_type(m) for m in methods]
        return [f'{m} (bead type = {b})' for m, b in zip(methods, bead_types)]
    

def timer(func: T) -> T:
    """
    Timing decorator for functions. If the function is a method of a class,
    the class must have a _verbose attribute which determines if timing
    information is printed. Otherwise, timing information is always printed.

    :param func: The function to time.
    :return: The wrapped function.
    """
    def wrapper(*args, **kwargs) -> Any:
        # The function could be a standalone function or a method of a class.
        # If it's a method, args[0] will be self. We need to check if _verbose 
        # attribute exists and if it's set to True.
        verbose = getattr(args[0], "_verbose", False) if args else True

        # if verbose is not True, don't print timing information.
        if not verbose:
            return func(*args, **kwargs)

        start_time = perf_counter()
        result = func(*args, **kwargs)
        end_time = perf_counter()
        elapsed_time = end_time - start_time
        print(Colours.colourise(
            f"{func.__name__}() took {elapsed_time:.2f} seconds to execute.",
            Colours.OKBLUE
        ))
        return result
    return wrapper


def get_nn_dict(cnn, structure, ix):
    """
    For the given atom indices, calculates the nearest neighbours using the
    crystalNN method.

    :param cnn: crystalNN object
    :param structure: pymatgen structure object
    :param ix: list of atom indices.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {i:cnn.get_nn_info(structure,i) for i in ix}
    

def _convert_site_to_index(site: str) -> int:
    """
    Convert a single site letter to index (where "a" = 0, "b" = 1, etc.).

    :param site: label 'a', 'b', etc. to convert into zero-indexing from 'a'.
    :return: index of site.
    """
    if 'a' <= site <= 'z':
        return ord(site) - ord('a')
    elif 'A' <= site <= 'Z':
        return ord(site) - ord('A')
    else:
        raise ValueError(f'Invalid character {site}. Input should only ' \
                         'contain alphabetical characters.')


def site_type_to_index(sites: Union[str, List[str]]) -> List[int]:
    """
    Convert site letters to indices (where "a" = 0, "b" = 1, etc.).

    :param sites: labels 'a', 'b', etc. to convert into zero-indexing from 'a'.
    :return: list of indices of sites.
    """
    if isinstance(sites, str):
        sites = ''.join(sites.split())
    elif sites is None:
        return []
    return [_convert_site_to_index(s) for s in sites]


def get_first_n_letters(n: int) -> List[str]:
    """
    Return the first n letters of the alphabet.
    
    :param n: number of letters to return.
    :return: list of letters.
    """
    if not 1 <= n <= 26:
        raise ValueError("n must be between 1 and 26")
    return list(ascii_lowercase[:n])


def parse_arguments(
    default: Union[float, None], 
    values: Union[float, List[float], Dict[str, float]], 
    keys: List[str]
) -> Dict[str, Any]:
    """
    Parses arguments that could be a single value, a list of values, or a 
    dictionary of values.

    :param default: the default value to use if a value is not provided.
    :param values: The provided values, which could be a single value, a list of 
        values, or a dictionary of values.
    :param keys: The list of keys.
    :return: dictionary mapping each key to its corresponding value.
    """
    if isinstance(values, dict):
        return {k: values.get(k, default) for k in keys}
    elif isinstance(values, (list, tuple)):
        if len(values) != len(keys):
            raise ValueError('The length of the list of values must match the' \
                ' number of site types.')
        return dict(zip(keys, values))
    elif values is None or isinstance(values, (int, float)):
        return {k: values if values is not None else default for k in keys}
    else:
        raise TypeError(f"Invalid type for values: {type(values)}.")
    

def are_arrays_equal(iter_arrays):
    """
    Checks if all arrays in an iterable are equal.

    This function receives an iterable of numpy arrays and checks whether they 
    are all equal both in terms of shape and content. The comparison is done 
    pairwise starting from the first array.

    :params iter_arrays: An iterable (e.g., list, tuple, etc.) of numpy arrays.
    :returns: True if all arrays are equal, False otherwise.
    :rasies StopIteration: An empty iterator was passed to the function.

    Examples:
    >>> are_arrays_equal([np.array([1, 2, 3]), np.array([1, 2, 3]), np.array([1, 2, 3])])
    True

    >>> are_arrays_equal([np.array([1, 2, 3]), np.array([1, 2, 4])])
    False
    """
    iter_arrays = iter(iter_arrays)
    try:
        first_array = next(iter_arrays)
    except StopIteration:
        return True
    return all(np.array_equal(first_array, array) for array in iter_arrays)



def round_to_precision(
    value: Union[float, np.ndarray], 
    precision: float = 1e-8,
):
    """ 
    Round a value (or array of values) to a given precision.

    :param value: The value to round.
    :param precision: The precision to round to.
    :return: The rounded value.
    """
    decimal_places = abs(int(log10(precision)))
    return np.round(value, decimal_places)


def crystal_toolkit_display(structure):
    """
    Display a structure using crystal toolkit.
    """
    try:
        # catch the " No module named 'phonopy' " warning.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from dash import Dash, html
            import crystal_toolkit.components as ctc
            from crystal_toolkit.settings import SETTINGS
    except ImportError:
        raise ImportError("Crystal toolkit is not installed.")
    
    # setup the app with the structure.
    app = Dash(assets_folder=SETTINGS.ASSETS_PATH)
    struct_comp = ctc.StructureMoleculeComponent(
        id="structure", struct_or_mol=structure
    )
    app.layout = html.Div([struct_comp.layout()])
    ctc.register_crystal_toolkit(app=app, layout=app.layout)

    # deploy the app.
    app.run()


def most_common_value(a_list: List[Any]) -> Any:
    """
    Returns the most common value in a list.
    
    :param a_list: The list to check.
    :return: The most common value.
    """
    return Counter(a_list).most_common(1)[0][0]