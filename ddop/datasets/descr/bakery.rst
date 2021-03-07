.. _bakery_dataset:

Bakery dataset
----------------

This is a real world dataset provided by a bakery. The dataset contains the demand for rolls, seeded rolls and
pretzels. Moreover, it stores a number of demand features. A description of targets and features is given below.


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
    - 'promotion_lastweek' whether ore not there was a promotion last week

    Note: The dataset also includes demand lag features as well as a column for the demand date.
    By default, those features are not included when loading the data. You can include them
    by setting the parameter `include_lag`/`include_date` to `True`.





