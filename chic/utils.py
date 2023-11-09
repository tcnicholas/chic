"""
30.07.23
@tcnicholas
Utility functions.
"""

import os
import re
import sys
import warnings
import contextlib
from math import log10
from pathlib import Path
from time import perf_counter
from collections import Counter
from string import ascii_lowercase
from typing import List, Callable, Any, Union, TypeVar, Dict, Tuple

import numpy as np
from ase.units import kcal, mol

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
    

def timer(func: Union[T, None] = None, *, colour: str = Colours.OKBLUE) -> T:
    """
    Timing decorator for functions. If the function is a method of a class,
    the class must have a _verbose attribute which determines if timing
    information is printed. Otherwise, timing information is always printed.

    :param func: The function to time.
    :param colour: The colour to use for the printed timing information.
    :return: The wrapped function.
    """
    def wrapper(*args, **kwargs) -> Any:
        verbose = getattr(args[0], "_verbose", False) if args else True
        if not verbose:
            return func(*args, **kwargs)

        start_time = perf_counter()
        result = func(*args, **kwargs)
        end_time = perf_counter()
        elapsed_time = end_time - start_time
        print(Colours.colourise(
            f"{func.__name__}() took {elapsed_time:.2f} seconds to execute.",
            colour
        ))
        return result

    if func is None:
        return lambda f: timer(f, colour=colour)
    return wrapper


def rename_file(file_path: str, new_filename: str) -> None:
    """
    Rename a file to a new filename using pathlib for platform independence.

    Arguments:
        file_path: The path to the file to be renamed.
        new_filename: The new filename.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        FileExistsError: If a file with the new filename already exists.
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError("The specified file does not exist")

    new_file_path = file_path.parent / new_filename
    if new_file_path.exists():
        raise FileExistsError("A file with the new filename already exists")

    file_path.rename(new_file_path)


def delete_files_except_suffixes(
    directory: Path, 
    suffixes: Union[str, List[str]]
) -> None:
    """
    Delete all files in a given directory except for ones with specified suffixes.

    Arguments:
        directory: The path to the directory whose files you want to delete.
        suffixes: Suffix or list of suffixes of the files you want to keep.
    """
    if not isinstance(directory, Path):
        directory = Path(directory)

    if not directory.is_dir():
        print("Error: The specified path is not a directory.")
        return

    # Handle single string suffix
    if isinstance(suffixes, str):
        suffixes = [suffixes]
    elif not isinstance(suffixes, list):
        print("Error: The suffixes parameter should be a string or a list.")
        return

    # Normalize suffixes to start with a dot (.)
    suffixes = [
        '.' + suffix if not suffix.startswith('.') else suffix
        for suffix in suffixes
    ]

    deleted_files = []
    for file in directory.glob('*'):
        if file.is_file() and file.suffix not in suffixes:
            file.unlink()
            deleted_files.append(file.name)

    if deleted_files:
        print(f"Deleted files: {', '.join(deleted_files)}")
    else:
        print("No files were deleted.")


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


def strip_number(x: str) -> str:
    """
    Remove number from atom label to get the symbol.

    :param x: atom label with number.
    :return: atom symbol without number.
    """
    return ''.join([s for s in x if not s.isdigit()])


def setattrs(_self, **kwargs):
    """
    Set attributes of a dataclass.
    """
    for k,v in kwargs.items():
        setattr(_self, k, v)
        

def replace_a_and_b(x: str) -> str:
    """
    Replace a and b with nothing.
    """
    return x.replace('a','').replace('b','')


def atom2str(atom: list):
    """
    Make LAMMPS-like atom string.

    :param atom: atom list.
    :return: LAMMPS-like atom string.
    """
    return "{:>10.0f} {:>10.0f} {:>10.0f} {:>20.5f} {:>20.10f} {:>20.10f} {:>20.10f}\n".format(*atom)


def sorted_directories(parent: Path) -> List:
    """
    Get all directories in a parent directory, sorted by name. Will attempt to
    convert the directory name to a float, and if this fails, will assign a
    value of infinity.

    :param parent: The parent directory.
    :return: A list of directories.
    """
    
    parent = Path(parent)
    directories = [d for d in parent.iterdir() if d.is_dir()]

    def dir_key(x: Path) -> Tuple[Union[float, str], str]:
        """
        Get the directory name as a float and string part.

        :param x: The directory.
        :return: The directory name as a tuple (float part, string part).
        """
        match = re.match(r"([+-]?\d*\.?\d*)(.*)", x.name)
        if match:
            num_part, str_part = match.groups()
            try:
                return float(num_part), str_part
            except ValueError:
                return float('inf'), str_part
        else:
            return float('inf'), x.name

    return sorted(directories, key=dir_key)


def kcalmol2eV(kcalmol: Union[float, np.ndarray]) -> np.ndarray:
    """
    Convert kcal/mol to eV.
    """
    return kcalmol * kcal  / mol


class NullIO:
    def write(self, *args, **kwargs):
        pass

    def read(self, *args, **kwargs):
        return ''

    def flush(self, *args, **kwargs):
        pass

    def close(self, *args, **kwargs):
        pass


@contextlib.contextmanager
def suppress_output():
    # Store original stdout and stderr
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr

    # Replace stdout and stderr with our "null" versions
    sys.stdout = NullIO()
    sys.stderr = NullIO()

    try:
        yield
    finally:
        # Restore original stdout and stderr
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr


def no_output(func):
    def wrapper(*args, **kwargs):
        with suppress_output():
            return func(*args, **kwargs)
    return wrapper


def mkdir(dirname: str) -> Path:
    """
    Generate a directory in one line.

    :param dirname: name of directory to make.
    :return: Path object for directory.
    """
    directory = Path(dirname)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def remove_non_letters(s: str) -> str:
    """
    Remove non-letter characters from a string.

    :param s: string to modify.
    :return: modified string.
    """
    return "".join((x for x in s if x.isalpha()))


def remove_symbols(s: str) -> str:
    """
    Remove non-alphanumeric characters from a string.

    :param s: string to modify.
    :return: modified string.
    """
    return re.sub(r'[^a-zA-Z0-9]', '', s)


def remove_uncertainties(s: str) -> float:
    """
    Remove uncertainty brackets from strings and return the float.

    :param s: string to convert to float.
    :return: float value of string.
    """
    try:
        # Note that the ending ) is sometimes missing. That is why the code has
        # been modified to treat it as optional. Same logic applies to lists.
        return float(re.sub(r"\(.+\)*", "", s))
    except TypeError:
        if isinstance(s, list) and len(s) == 1:
            return float(re.sub(r"\(.+\)*", "", s[0]))
    except ValueError as exc:
        if s.strip() == ".":
            return 0
        raise exc
    raise ValueError(f"{s} cannot be converted to float")