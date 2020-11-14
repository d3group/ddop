.. _yaz_dataset:

YAZ dataset
----------------

This is a real world dataset provided by a bakery. The dataset contains the demand for rolls, seeded rolls and
pretzels. Moreover, it stores a number of demand features. A description of targets and features is given below.


**Dataset Characteristics:**

  :Number of Instances: 1155

  :Number of Targets: 3

  :Number of Features: 6

  :Target Information:
    - 'roll' the demand for calamari
    - 'seeded_roll' the demand for fish
    - 'pretzel' the demand for shrimps

  :Feature Information:
    - 'weekday' the day of the week,
    - 'month' the month of the year,
    - 'year' the year,
    - 'is_holiday' whether it is a national holiday,
    - 'rainfall' the amount of rainfall,
    - 'temperature' the outdoor temperature,

Note: The dataset also includes a column for the demand date. By default, the date
is not included when loading the data. You can include it by setting the parameter
`include_date` to `True`.





