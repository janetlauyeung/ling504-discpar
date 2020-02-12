import pandas as pd


def coerce_numerical(data):
    # Some numerical columns are object because they contain missing values
    # You may want to convert them to numerical and fille NAs somehow
    num_with_missing = [
        'First-QueueDist-To-Begin',
        'Top2-StackDist-To-End',
        'Top2-StackLength-EDU',
        'Top1-StackLength-EDU',
        'Top1-StacknEDUs'
    ]

    for col in num_with_missing:
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(value=-1)


def suppress_rare(data):
    # Depending on the approach you take you may want to handle rare values
    # which might be absent from test
    for col in ['Top12-StackSType', 'Stack-QueueSType']:
        data.loc[data[col].value_counts()[data[col]].values < 20, col] = "_"


def clean(data):
    coerce_numerical(data)
    suppress_rare(data)
    return data
