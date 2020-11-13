.. _yaz_dataset:

YAZ dataset
----------------

This is a real world dataset from Yaz. Yaz is a fast casual restaurant in Stuttgart providing good service
and food at short waiting times. The dataset contains the demand for the main ingredients for the meals at YAZ.
Moreover, it stores a number of demand features. These features include information about the day, month, year,
lag demand, weather conditions and more. A full description of targets and features is given below.


**Dataset Characteristics:**

  :Number of Instances: 760

  :Number of Targets: 7

  :Number of Features: 10

  :Target Information:
    - 'CALAMARI' the demand for calamari
    - 'FISH' the demand for fish
    - 'SHRIMP' the demand for shrimps
    - 'CHICKEN' the demand for chicken
    - 'KOEFTE' the demand for koefte
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

Note: The dataset also includes 193 weather and demand lag features as well as a column for the demand date.
By default, those features are not included when loading the data. You can include them by setting the parameter
``include_lag``/``include_date`` to ``True``.





