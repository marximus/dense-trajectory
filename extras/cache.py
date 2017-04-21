import os


def get_cache_dir(root_dir, params):
    cache_dir = os.path.join(root_dir, _dict_to_string(params))
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
    return cache_dir


def _dict_to_string(d):
    """Return a string representation of the key/value pairs in dictionary."""
    s = '_'.join(['{}-{}'.format(k, d[k]) for k in sorted(d)])
    s = s.replace(' ', '')
    return s

