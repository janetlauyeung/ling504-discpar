import pandas as pd, numpy as np
import io
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import matplotlib.pyplot as plt

# load the dataset
facts = pd.read_csv("rst_transitions.tab", sep="\t", quoting=3)

facts.info()


facts.head()



# Clean
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


data = facts.copy(deep=True)
data = clean(data)



# Augment
def augment(data):
    # If you want to engineer some table features
    # before creating a dev partition, this is the place!
    data["First-Queue-Len"] = data["First-Queue"].str.len()
    return data


data = augment(data)
data = data.sample(frac=1, random_state=42)


# Split
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_idx, test_idx in splitter.split(data, data["label"]):
    train = data.loc[train_idx]
    test = data.loc[test_idx]

# We have this many examples, stratified by number of children
print("Train, test sizes:")
print(train.shape, test.shape)


# Preprocess
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin


# This is effectively a 'do nothing transformer', we may need it below
class IdentityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, input_array, y=None):
        return self

    def transform(self, input_array, y=None):
        return input_array * 1



# You can engineer your lexical vocabulary here if needed
# Get a whitelist based on frequency threshold

# Some possible resources
# "vocab_freqs.tab" - unigram and bigram frequencies in the training corpus
# "vocab_docs.tab" -  unigram and bigram frequencies by document; number of lines per items = document frequency
# "pdtb_conn.tab" - frequency of items flagged as discourse connectives in PDTB (incl. multiword expressions)

lex_feats = ["Top2-Stack", "Top1Span", "First-Queue"]

# Store a vocabulary per feature
vocabs = {}
lex_vectorizers = {}
for feat in lex_feats:
    cvec = CountVectorizer(lowercase=False,
                           ngram_range=(1, 2),
                           # vocabulary=whitelist,   # You can work with your own whitelist
                           max_features=1000,  # Or work with the top 1000 most frequent items, or...
                           token_pattern=u"(?u)\\b\\S+\\b",  # Use these settings if you want to keep punctuation
                           analyzer="word")
    cvec.fit(train[feat])
    vocabs[feat] = cvec.get_feature_names()
    lex_vectorizers[feat] = cvec

    print(feat)
    print(vocabs[feat][:100])


categorical_features = ['Top12-StackXML', 'Stack-QueueSType', "Stack", "genre", "Stack-QueueSameSent",
                        "Top12-StackSameSent",
                        'Top12-StackSameSent', 'Stack-QueueXML', 'Top12-StackSType',
                        "Top12-StackDir", "Stack-QueueDir", "First-QueueEduFunc", "Top1SpanEduFunc"]
numeric_features = ['First-QueueDist-To-Begin', 'Top2-StackLength-EDU', 'Top1-StackLength-EDU', 'Top1-StacknEDUs']
scale_features = ['Top2-StackDist-To-End', 'First-Queue-Len']
text_features = ['First-Queue', 'Top1Span', 'Top2-Stack']


def encode(data, one_hot=False, ordinal=False):
    numeric_transformer = Pipeline(steps=[
        ('identity', IdentityTransformer())
    ])

    scale_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_steps = []
    categorical_steps += [('onehot', OneHotEncoder(handle_unknown='ignore'))] if one_hot else []
    categorical_steps += [('ordinal', OrdinalEncoder())] if ordinal else []
    categorical_transformer = Pipeline(steps=categorical_steps)

    text_transformer = ColumnTransformer(transformers=[
        ('count', lex_vectorizers["First-Queue"], "First-Queue"),
        ('count2', lex_vectorizers["Top1Span"], "Top1Span"),
        ('count3', lex_vectorizers["First-Queue"], "Top2-Stack"),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('sca', scale_transformer, scale_features),
            ('text', text_transformer, text_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    return preprocessor.fit_transform(data)


encoded_oh = encode(train, one_hot=True)
encoded_ord = encode(train, ordinal=True)

# Learn
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

clf = LinearSVC()
clf.fit(encoded_ord, train["label"])
"OK"

from sklearn.metrics import classification_report

X_test = encode(test)

preds = clf.predict(X_test)

print(classification_report(test["label"], preds))

# Useful function for retreiving feature names

def get_column_names_from_ColumnTransformer(column_transformer):
    col_name = []
    for transformer_in_columns in column_transformer.transformers_[
                                  :-1]:  # the last transformer is ColumnTransformer's 'remainder'
        raw_col_name = transformer_in_columns[2]
        if isinstance(transformer_in_columns[1], Pipeline):
            transformer = transformer_in_columns[1].steps[-1][1]
        else:
            transformer = transformer_in_columns[1]
        try:
            names = transformer.get_feature_names()
        except AttributeError:  # if no 'get_feature_names' function, use raw column name
            names = raw_col_name
        if isinstance(names, np.ndarray):  # eg.
            col_name += names.tolist()
        elif isinstance(names, list):
            col_name += names
        elif isinstance(names, str):
            col_name.append(names)
    return col_name


names = get_column_names_from_ColumnTransformer(preprocessor)
print(names[:50])

