import pandas as pd
import numpy as np
from os.path import dirname, join
from sklearn.utils import Bunch


def load_yaz(include_prod=None, include_lag=False, include_date=False, encode_date_features=False,
             categorical_to_continuous=False, return_X_y=False):
    """Load and return the YAZ dataset

    Yaz is a fast casual restaurant in Stuttgart providing good service and food at short waiting times.
    The dataset contains the demand for the main ingredients for the meals at YAZ. Moreover, it stores a
    number of demand features. A description of targets and features is given below.

    **Dataset Characteristics:**

        :Number of Instances: 760

        :Number of Attributes: 10

        :Number of Targets: 7

        :Target Information:
            - 'CALAMARI' the demand for calamari
            - 'FISH' the demand for fish
            - 'SHRIMP' the demand for shrimps
            - 'CHICKEN' the demand for koefte
            - 'LAMB' the demand for lamb
            - 'STEAK' the demand for steak

        :Feature Information:
            - 'WEEKDAY' the day of the week,
            - 'MONTH' the month of the year,
            - 'YEAR' the year,
            - 'ISHOLIDAY' whether it is a national holiday,
            - 'WEEKEND' whether it is weekend,
            - 'WIND' the wind force,
            - 'CLOUDS' the cloudiness degree,
            - 'RAINFALL' the amount of rainfall,
            - 'HOURS_OF_SUNSHINE' the number of sunshine hours,
            - 'TEMPERATURE' the outdoor temperature,

        Note: The dataset also includes 194 weather and demand lag features as well as a column for the demand date.
        By default, those features are not included when loading the data. You can include them
        by setting the parameter `include_lag`/`include_date` to `True`.

    Parameters
    ----------
    include_prod : 1d array or list , default=None
        List of products to include. Valid products are {"STEAK", "CHICKEN", "KOEFTE", "LAMB", "FISH", "SHRIMP",
        "CALAMARI"}. If None, all products are included
    include_lag : bool, default=False
        Whether to include lag features
    include_date : bool, default=False
        Whether to include the demand date
    encode_date_features : bool, default=False
        Whether to one hot encode column WEEKDAY, MONTH, YEAR
    categorical_to_continuous : bool, default=False
        Whether to convert categorical columns (WEEKDAY, MONTH, YEAR) to continuous.
        Will only be applied if `encode_date_features=False`
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
        data = data.drop('DEMAND_DATE', axis=1)

    feature_terms_to_drop = []
    targets_to_drop = []

    if not include_lag:
        for term in ["_T1", "_T2", "_T3", "_T4", "_T5", "_T6", "_T7", "_7D_", "_W2", "_W3", "_W4"]:
            feature_terms_to_drop.append(term)

    if include_prod is not None:

        products = ["STEAK", "CHICKEN", "KOEFTE", "LAMB", "FISH", "SHRIMP", "CALAMARI"]

        if not np.any([prod in products for prod in include_prod]):
            raise ValueError(
                "No valid product in include_prod. If you specify this parameter, please select at least one valid "
                "product. Supported are %s" % (list(products)))

        if "LAMB" not in include_prod:
            feature_terms_to_drop.append("LAMB")
            targets_to_drop.append("LAMB")

        if "STEAK" not in include_prod:
            feature_terms_to_drop.append("STEAK")
            targets_to_drop.append("STEAK")

        if "KOEFTE" not in include_prod:
            feature_terms_to_drop.append("KOEFTE")
            targets_to_drop.append("KOEFTE")

        if "CHICKEN" not in include_prod:
            feature_terms_to_drop.append("CHICKEN")
            targets_to_drop.append("CHICKEN")

        if "SHRIMP" not in include_prod:
            feature_terms_to_drop.append("SHRIMP")
            targets_to_drop.append("SHRIMP")

        if "FISH" not in include_prod:
            feature_terms_to_drop.append("FISH")
            targets_to_drop.append("FISH")

        if "CALAMARI" not in include_prod:
            feature_terms_to_drop.append("CALAMARI")
            targets_to_drop.append("CALAMARI")

    for col in data.columns:
        if np.any([term in col for term in feature_terms_to_drop]):
            data = data.drop(col, 1)

    target = target.drop(targets_to_drop, axis=1)

    frame = pd.concat([data, target], axis=1)

    n_features = data.shape[0]
    n_targets = data.shape[1]

    if encode_date_features:
        data = pd.get_dummies(data, ["WEEKDAY", "MONTH", "YEAR"])

    if not encode_date_features and categorical_to_continuous:
        data['WEEKDAY']=data['WEEKDAY'].apply(_day_to_continuouse)
        data['MONTH'] = data['MONTH'].apply(_month_to_continuouse)
        data['YEAR'] = data['YEAR'].apply(int)

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

