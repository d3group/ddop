import pandas as pd
import numpy as np
from os.path import dirname, join
from sklearn.utils import Bunch


def load_yaz(include_date=False, one_hot_encoding=False,
             label_encoding=False, return_X_y=False):
    """Load and return the YAZ dataset

    Yaz is a fast casual restaurant in Stuttgart providing good service and food at short waiting times.
    The dataset contains the demand for the main ingredients at YAZ. Moreover, it stores a
    number of demand features. A description of targets and features is given below.

    **Dataset Characteristics:**

        :Number of Instances: 765

        :Number of Targets: 7

        :Number of Features: 12

        :Target Information:
            - 'calamari' the demand for calamari
            - 'fish' the demand for fish
            - 'shrimp' the demand for shrimps
            - 'chicken' the demand for chicken
            - 'koefte' the demand for koefte
            - 'lamb' the demand for lamb
            - 'steak' the demand for steak

        :Feature Information:
            - 'date' the date,
            - 'weekday' the day of the week,
            - 'month' the month of the year,
            - 'year' the year,
            - 'is_holiday' whether or not it is a national holiday,
            - 'is_closed' whether or not the restaurant is closed,
            - 'weekend' whether or not it is weekend,
            - 'wind' the wind force,
            - 'clouds' the cloudiness degree,
            - 'rain' the amount of rain,
            - 'sunshine' the sunshine hours,
            - 'temperature' the outdoor temperature,

    Parameters
    ----------
    include_date : bool, default=False
        Whether to include the demand date
    one_hot_encoding : bool, default=False
        Whether to one hot encode categorical features
    label_encoding : bool, default=False
        Whether to convert categorical columns (weekday, month, year) to continuous.
        Will only be applied if `one_hot_encoding=False`
    return_X_y : bool, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : sklearn Bunch
        Dictionary-like object, with the following attributes.

        data : Pandas DataFrame of shape (765, n_features)
            The data matrix.
        target: Pandas DataFrame of shape (765, n_targets)
            The target values.
        n_features: int
            The number of features included
        n_targets: int
            The number of target variables included
        DESCR: str
            The full description of the dataset.
        data_filename: str
            The path to the location of the data.
        target_filename: str
            The path to the location of the target.

    (data, target) : tuple if ``return_X_y`` is True

    Examples
    ---------
    >>> from ddop.datasets import load_yaz
    >>> X, y = load_yaz(return_X_y=True)
    >>> print(X.shape)
        (765, 11)
    """

    module_path = dirname(__file__)
    base_dir = join(module_path, 'data')
    data_filename = join(base_dir, 'yaz_data.csv')
    data = pd.read_csv(data_filename)
    target_filename = join(base_dir, 'yaz_target.csv')
    target = pd.read_csv(target_filename)

    with open(join(module_path, 'descr', 'yaz.rst')) as rst_file:
        fdescr = rst_file.read()

    if not include_date:
        data = data.drop('date', axis=1)

    n_features = data.shape[0]
    n_targets = data.shape[1]

    if one_hot_encoding:
        data = pd.get_dummies(data, columns=["weekday", "month", "year"])

    elif not one_hot_encoding and label_encoding:
        data['weekday'] = data['weekday'].apply(_day_to_continuouse)
        data['month'] = data['month'].apply(_month_to_continuouse)

    else:
        data['year'] = data['year'].apply(str)

    if return_X_y:
        return data, target

    return Bunch(data=data,
                 target=target,
                 n_features=n_features,
                 n_targets=n_targets,
                 DESCR=fdescr,
                 data_filename=data_filename,
                 target_filename=target_filename)


def load_bakery(include_date=False, one_hot_encoding=False,
             label_encoding=False, return_X_y=False):
    """Load and return the bakery dataset

    The bakery dataset contains the demand for a number of products from different stores. Moreover, it stores a
    number of demand features. A description of targets and features is given below.

    **Dataset Characteristics:**

        :Number of Instances: 127575

        :Number of Targets: 1

        :Number of Features: 13

        :Target Information:
            - 'demand' the corresponding demand observation

        :Feature Information:
            - 'date' the date
            - 'weekday' the day of the week,
            - 'month' the month of the year,
            - 'year' the year,
            - 'is_holiday' whether or not it is a national holiday,
            - 'is_holiday_next2days' whether or not it is a national holiday in the next two days,
            - 'is_schoolholiday' whether or not it is a school holiday,
            - 'store' the store id,
            - 'product' the product id,
            - 'rain' the amount of rain,
            - 'temperature' the average temperature in °C,
            - 'promotion_currentweek' whether or not there is a promotion this week
            - 'promotion_lastweek' whether there was a promotion last week

    Parameters
    ----------
    include_date : bool, default=False
        Whether to include the demand date
    one_hot_encoding : bool, default=False
        Whether to one hot encode categorical features
    label_encoding : bool, default=False
        Whether to convert categorical columns (weekday, month, year) to continuous.
        Will only be applied if `one_hot_encoding=False`
    return_X_y : bool, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : sklearn Bunch
        Dictionary-like object, with the following attributes.

        data : Pandas DataFrame of shape (127575, n_features)
            The data matrix.
        target: Pandas DataFrame of shape (127575, n_targets)
            The target values.
        n_features: int
            The number of features included
        n_targets: int
            The number of target variables included
        DESCR: str
            The full description of the dataset.
        data_filename: str
            The path to the location of the data.
        target_filename: str
            The path to the location of the target.
    (data, target) : tuple if ``return_X_y`` is True

    Examples
    ----------
    >>> from ddop.datasets import load_bakery
    >>> X, y = load_bakery(return_X_y=True)
    >>> print(X.shape)
        (127575, 12)
    """

    module_path = dirname(__file__)
    base_dir = join(module_path, 'data')
    data_filename = join(base_dir, 'bakery_data.csv')
    data = pd.read_csv(data_filename)
    target_filename = join(base_dir, 'bakery_target.csv')
    target = pd.read_csv(target_filename)

    with open(join(module_path, 'descr', 'bakery.rst')) as rst_file:
        fdescr = rst_file.read()

    if not include_date:
        data = data.drop('date', axis=1)

    n_features = data.shape[0]
    n_targets = data.shape[1]

    if one_hot_encoding:
        data = pd.get_dummies(data, columns=["weekday", "month", "year"])

    elif not one_hot_encoding and label_encoding:
        data['weekday'] = data['weekday'].apply(_day_to_continuouse)
        data['month'] = data['month'].apply(_month_to_continuouse)

    else:
        data['year'] = data['year'].apply(str)

    if return_X_y:
        return data, target

    return Bunch(data=data,
                 target=target,
                 n_features=n_features,
                 n_targets=n_targets,
                 DESCR=fdescr,
                 data_filename=data_filename,
                 target_filename=target_filename)


def load_SID(include_date=False, one_hot_encoding=False,
             label_encoding=False, return_X_y=False):

    """Load and return the store item demand dataset.

    **Dataset Characteristics:**

        :Number of Instances: 887284

        :Number of Targets: 1

        :Number of Features: 6

        :Target Information:
            - 'demand' the corresponding demand observation

        :Feature Information:
            - 'date' the date
            - 'weekday' the day of the week,
            - 'month' the month of the year,
            - 'year' the year,
            - 'store' the store id,
            - 'item' the item id
    Parameters
    ----------
    include_date : bool, default=False
        Whether to include the demand date
    one_hot_encoding : bool, default=False
        Whether to one hot encode categorical features
    label_encoding : bool, default=False
        Whether to convert categorical columns (weekday, month, year) to continuous.
        Will only be applied if `one_hot_encoding=False`
    return_X_y : bool, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : sklearn Bunch
        Dictionary-like object, with the following attributes.

        data : Pandas DataFrame of shape (887284, n_features)
            The data matrix.
        target: Pandas DataFrame of shape (887284, n_targets)
            The target values.
        n_features: int
            The number of features included
        n_targets: int
            The number of target variables included
        DESCR: str
            The full description of the dataset.
        data_filename: str
            The path to the location of the data.
        target_filename: str
            The path to the location of the target.
    (data, target): tuple if ``return_X_y`` is True

    Notes
    -----
    The store item demand dataset was published within a demand forecasting challenge on kaggle [1]

    References
    ----------
    .. [1] https://www.kaggle.com/c/demand-forecasting-kernels-only/overview

    Examples
    --------
    >>> from ddop.datasets import load_SID
    >>> X, y = load_SID(return_X_y=True)
    >>> print(X.shape)
        (887284, 5)
    """

    module_path = dirname(__file__)
    base_dir = join(module_path, 'data')
    data_filename = join(base_dir, 'SID_data.csv')
    data = pd.read_csv(data_filename)
    target_filename = join(base_dir, 'SID_target.csv')
    target = pd.read_csv(target_filename)

    with open(join(module_path, 'descr', 'SID.rst')) as rst_file:
        fdescr = rst_file.read()

    if not include_date:
        data = data.drop('date', axis=1)

    n_features = data.shape[0]
    n_targets = data.shape[1]

    if one_hot_encoding:
        data = pd.get_dummies(data, columns=["weekday", "month", "year"])

    elif not one_hot_encoding and label_encoding:
        data['weekday'] = data['weekday'].apply(_day_to_continuouse)
        data['month'] = data['month'].apply(_month_to_continuouse)

    else:
        data['year'] = data['year'].apply(str)

    if return_X_y:
        return data, target

    return Bunch(data=data,
                 target=target,
                 n_features=n_features,
                 n_targets=n_targets,
                 DESCR=fdescr,
                 data_filename=data_filename,
                 target_filename=target_filename)


def _month_to_continuouse(x):
    if x=='JAN':
        return 1
    elif x=='FEB':
        return 2
    elif x=='MAR':
        return 3
    elif x=='APR':
        return 4
    elif x=='MAY':
        return 5
    elif x=='JUN':
        return 6
    elif x=='JUL':
        return 7
    elif x=='AUG':
        return 8
    elif x=='SEP':
        return 9
    elif x=='OCT':
        return 10
    elif x=='NOV':
        return 11
    else:
        return 12


def _day_to_continuouse(x):
    if x=='MON':
        return 1
    elif x=='TUE':
        return 2
    elif x=='WED':
        return 3
    elif x=='THU':
        return 4
    elif x=='FRI':
        return 5
    elif x=='SAT':
        return 6
    else:
        return 7