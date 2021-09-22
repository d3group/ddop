.. _SID_dataset:

SID dataset
-------------

This dataset contains 5 zears of store-item demand data

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

    Note: By default the date feature is not included when loading the data. You can include it
    by setting the parameter `include_date` to `True`.





