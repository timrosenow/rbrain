"""
Exception handling for rbrain.
"""


class ConfigFileNotFoundException(Exception):
    """
    Exception when loading a config file that doesn't exist.
    """
    pass


class DatasetNotLoadedException(Exception):
    """
    Exception when trying to perform an action when the dataset is not yet loaded.
    """
    pass


class InvalidDatasetException(Exception):
    """
    Exception when trying to operate on an invalid dataset.
    """


class NNModelNotFoundException(Exception):
    """
    Exception when trying to operate on a neural net (TF) model that has not yet been created/loaded.
    """
    