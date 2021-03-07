.. _yaz_dataset:

YAZ dataset
----------------

This is a real world dataset from Yaz. Yaz is a fast casual restaurant in Stuttgart providing good service
and food at short waiting times. The dataset contains the demand for the main ingredients at YAZ.
Moreover, it stores a number of demand features. These features include information about the day, month, year,
lag demand, weather conditions and more. A full description of targets and features is given below.


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
        - 'is_holiday' whether it is a national holiday,
        - 'weekend' whether it is weekend,
        - 'wind' the wind force,
        - 'clouds' the cloudiness degree,
        - 'rainfall' the amount of rainfall,
        - 'sunshine' the sunshine hours,
        - 'temperature' the outdoor temperature,

    Note: The dataset also includes demand lag features as well as a column for the demand date.
    By default, those features are not included when loading the data. You can include them
    by setting the parameter `include_lag`/`include_date` to `True`.





