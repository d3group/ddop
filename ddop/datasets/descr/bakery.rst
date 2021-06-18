.. _bakery_dataset:

Bakery dataset
----------------

The bakery dataset contains the demand for a number of products from different stores. Moreover, it stores a number of
demand features. A description of targets and features is given below.


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

    Note: By default the date feature is not included when loading the data. You can include it
    by setting the parameter `include_date` to `True`.



