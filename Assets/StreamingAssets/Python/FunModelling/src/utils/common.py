import os

def recursive_mkdir(folder_path:str)->None:
    """
    Recursively create the folders leading to folder_path

    Arguments
    ---------
    - folder_path: (str)
      path to folder to create as well as its ancestors
    """
    parent_folder_path = os.path.dirname(folder_path)
    if not (os.path.exists(parent_folder_path)) and parent_folder_path != "":
        recursive_mkdir(parent_folder_path)
        if not(os.path.exists(folder_path)):
            os.makedirs(folder_path)
    else:
        if not(os.path.exists(folder_path)):
            os.makedirs(folder_path)

def flatten(dictionary, parent_key='', separator='_', keep_parent_key=True):
    """
    Flatten a nested dict into a single-level dict.

    Parameters
    ----------
    dictionary : dict
    parent_key : str, default ''
        Prefix for the keys (used in the recursion).
    separator : str, default '_'
        Joins parent and child keys.
    keep_parent_key : bool, default True
        If False, only the innermost key is kept (later duplicates overwrite earlier ones).

    Examples
    --------
    >>> flatten({"a": {"b": 1}, "c": 2})
    {'a_b': 1, 'c': 2}
    >>> flatten({"a": {"b": 1}}, keep_parent_key=False)
    {'b': 1}
    """
    items = []
    for key, value in dictionary.items():
        if keep_parent_key:
            new_key = parent_key + separator + key if parent_key else key
        else:
            new_key = key

        try:
            items.extend(flatten(value, new_key, separator=separator, keep_parent_key=keep_parent_key).items())
        except:
            items.append((new_key, value))
    return dict(items)
    