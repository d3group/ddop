import pandas as pd
import numpy as np
from os.path import dirname, join
from sklearn.utils import Bunch


def load_yaz(include_prod=None, include_lag=False, include_date=False, one_hot_encoding=False,
             label_encoding=False, return_X_y=False):
    """Load and return the YAZ dataset

    Yaz is a fast casual restaurant in Stuttgart providing good service and food at short waiting times.
    The dataset contains the demand for the main ingredients at YAZ. Moreover, it stores a
    number of demand features. A description of targets and features is given below.

    **Dataset Characteristics:**

        :Number of Instances: 760

        :Number of Targets: 7

        :Number of Features: 10

        :Target Information:
            - 'calamari' the demand for calamari
            - 'fish' the demand for fish
            - 'shrimp' the demand for shrimps
            - 'chicken' the demand for chicken
            - 'koefte' the demand for koefte
            - 'lamb' the demand for lamb
            - 'steak' the demand for steak

        :Feature Information:
            - 'weekday' the day of the week,
            - 'month' the month of the year,
            - 'year' the year,
            - 'is_holiday' whether or not it is a national holiday,
            - 'weekend' whether or not it is weekend,
            - 'wind' the wind force,
            - 'clouds' the cloudiness degree,
            - 'rainfall' the amount of rainfall,
            - 'sunshine' the sunshine hours,
            - 'temperature' the outdoor temperature,

        Note: The dataset also includes demand lag features as well as a column for the demand date.
        By default, those features are not included when loading the data. You can include them
        by setting the parameter `include_lag`/`include_date` to `True`.

    Parameters
    ----------
    include_prod : 1d array or list , default=None
        List of products to include. Valid products are {"steak", "chicken", "koefte", "lamb", "fish", "shrimp",
        "calamari"}. If None, all products are included
    include_lag : bool, default=False
        Whether to include lag features
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

        data : Pandas DataFrame of shape (760, n_features)
            The data matrix.
        target: Pandas DataFrame of shape (760, n_targets)
            The target values.
        frame: pandas DataFrame of shape (760, n_features+n_targets)
            Only present when `as_frame=True`. Pandas DataFrame with `data` and
            `target`.
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

    Examples:
    ----------
    >>> from ddop.datasets import load_yaz
    >>> X, y = load_yaz(return_X_y=True)
    >>> print(X.shape)
        (760, 10)
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

    feature_terms_to_drop = []
    targets_to_drop = []

    if not include_lag:
        for term in ["_t1", "_t2", "_t3", "_t4", "_t5", "_t6", "_t7", "_w2", "_w3", "_w4"]:
            feature_terms_to_drop.append(term)

    if include_prod is not None:

        products = ["steak", "chicken", "koefte", "lamb", "fish", "shrimp", "calamari"]

        if not np.any([prod in products for prod in include_prod]):
            raise ValueError(
                "No valid product in include_prod. If you specify this parameter, please select at least one valid "
                "product. Supported are %s" % (list(products)))

        if "lamb" not in include_prod:
            feature_terms_to_drop.append("lamb")
            targets_to_drop.append("lamb")

        if "steak" not in include_prod:
            feature_terms_to_drop.append("steak")
            targets_to_drop.append("steak")

        if "koefte" not in include_prod:
            feature_terms_to_drop.append("koefte")
            targets_to_drop.append("koefte")

        if "chicken" not in include_prod:
            feature_terms_to_drop.append("chicken")
            targets_to_drop.append("chicken")

        if "shrimp" not in include_prod:
            feature_terms_to_drop.append("shrimp")
            targets_to_drop.append("shrimp")

        if "fish" not in include_prod:
            feature_terms_to_drop.append("fish")
            targets_to_drop.append("fish")

        if "calamari" not in include_prod:
            feature_terms_to_drop.append("calamari")
            targets_to_drop.append("calamari")

    for col in data.columns:
        if np.any([term in col for term in feature_terms_to_drop]):
            data = data.drop(col, 1)

    target = target.drop(targets_to_drop, axis=1)

    n_features = data.shape[0]
    n_targets = data.shape[1]

    if one_hot_encoding:
        data = pd.get_dummies(data, columns=["weekday", "month", "year"])

    if not one_hot_encoding and label_encoding:
        data['weekday'] = data['weekday'].apply(_day_to_continuouse)
        data['month'] = data['month'].apply(_month_to_continuouse)
        data['year'] = data['year'].apply(int)

    frame = pd.concat([data, target], axis=1)

    if return_X_y:
        return data, target

    return Bunch(data=data,
                 target=target,
                 frame=frame,
                 n_features=n_features,
                 n_targets=n_targets,
                 DESCR=fdescr,
                 data_filename=data_filename,
                 target_filename=target_filename)


def load_bakery(include_prod=None, include_lag=False, include_date=False, one_hot_encoding=False,
             label_encoding=False, return_X_y=False):
    """Load and return the bakery dataset

    The bakery dataset contains the demand for rolls, seeded rolls and pretzels. Moreover, it stores a
    number of demand features. A description of targets and features is given below.

    **Dataset Characteristics:**

        :Number of Instances: 1155

        :Number of Targets: 3

        :Number of Features: 9

        :Target Information:
            - 'roll' the demand for rolls
            - 'seeded_roll' the demand for seeded rolls
            - 'pretzel' the demand for pretzels

        :Feature Information:
            - 'weekday' the day of the week,
            - 'month' the month of the year,
            - 'year' the year,
            - 'is_holiday' whether or not it is a national holiday
            - 'is_schoolholiday' whether or not it is a school holiday,
            - 'rainfall' the amount of rainfall,
            - 'temperature' the outdoor temperature,
            - 'promotion_currentweek' whether or not there is a promotion this week
            - 'promotion_lastweek' whether there was a promotion last week

        Note: The dataset also includes demand lag features as well as a column for the demand date.
        By default, those features are not included when loading the data. You can include them
        by setting the parameter `include_lag`/`include_date` to `True`.

    Parameters
    ----------
    include_prod : 1d array or list , default=None
        List of products to include. Valid products are {"roll", "seeded_roll", "pretzel"}.
        If None, all products are included
    include_lag : bool, default=False
        Whether to include lag features
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

        data : Pandas DataFrame of shape (1155, n_features)
            The data matrix.
        target: Pandas DataFrame of shape (1155, n_targets)
            The target values.
        frame: pandas DataFrame of shape (1155, n_features+n_targets)
            Only present when `as_frame=True`. Pandas DataFrame with `data` and
            `target`.
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

    Examples:
    ----------
    >>> from ddop.datasets import load_bakery
    >>> X, y = load_bakery(return_X_y=True)
    >>> print(X.shape)
        (1155, 9)
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

    feature_terms_to_drop = []
    targets_to_drop = []

    if not include_lag:
        for term in ["_t1", "_t2", "_t3", "_t4", "_t5", "_t6", "_t7", "_w2", "_w3", "_w4"]:
            feature_terms_to_drop.append(term)
    else:
        data.drop(data.index[:28], inplace=True)
        data.reset_index(drop=True, inplace=True)
        target.drop(data.index[:28], inplace=True)
        target.reset_index(drop=True, inplace=True)

    if include_prod is not None:

        products = ["roll", "seeded_roll", "pretzel"]

        if not np.any([prod in products for prod in include_prod]):
            raise ValueError(
                "No valid product in include_prod. If you specify this parameter, please select at least one valid "
                "product. Supported are %s" % (list(products)))

        if "roll" not in include_prod:
            targets_to_drop.append("roll")
            feature_terms_to_drop.append("roll")

        if "seeded_roll" not in include_prod:
            targets_to_drop.append("seeded_roll")
            feature_terms_to_drop.append("seeded_roll")

        if "pretzel" not in include_prod:
            targets_to_drop.append("pretzel")
            feature_terms_to_drop.append("pretzel")

    for col in data.columns:
        if np.any([term in col for term in feature_terms_to_drop]):
            data = data.drop(col, 1)

    target = target.drop(targets_to_drop, axis=1)

    n_features = data.shape[0]
    n_targets = data.shape[1]

    if one_hot_encoding:
        data = pd.get_dummies(data, columns=["weekday", "month", "year"])

    if not one_hot_encoding and label_encoding:
        data['weekday'] = data['weekday'].apply(_day_to_continuouse)
        data['month'] = data['month'].apply(_month_to_continuouse)
        data['year'] = data['year'].apply(int)

    frame = pd.concat([data, target], axis=1)

    if return_X_y:
        return data, target

    return Bunch(data=data,
                 target=target,
                 frame=frame,
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

