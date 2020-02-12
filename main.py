from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from clean import clean
from augment import augment

lex_feats = ["Top2-Stack", "Top1Span", "First-Queue"]
categorical_features = ['Top12-StackXML', 'Stack-QueueSType', "Stack", "genre", "Stack-QueueSameSent",
                        "Top12-StackSameSent",
                        'Top12-StackSameSent', 'Stack-QueueXML', 'Top12-StackSType',
                        "Top12-StackDir", "Stack-QueueDir", "First-QueueEduFunc", "Top1SpanEduFunc"]
numeric_features = ['First-QueueDist-To-Begin', 'Top2-StackLength-EDU', 'Top1-StackLength-EDU', 'Top1-StacknEDUs']
scale_features = ['Top2-StackDist-To-End', 'First-Queue-Len']
text_features = ['First-Queue', 'Top1Span', 'Top2-Stack']

# This is effectively a 'do nothing transformer', we may need it below
class IdentityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, input_array, y=None):
        return self

    def transform(self, input_array, y=None):
        return input_array * 1

def split(data):
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in splitter.split(data, data["label"]):
        train = data.loc[train_idx]
        test = data.loc[test_idx]
    return train, test

def make_preprocessor(data, one_hot=False, ordinal=False):
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
        cvec.fit(data[feat])
        vocabs[feat] = cvec.get_feature_names()
        lex_vectorizers[feat] = cvec

        print(feat)
        print(vocabs[feat][:100])

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

    return ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('sca', scale_transformer, scale_features),
            ('text', text_transformer, text_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )


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


def main():
    facts = pd.read_csv("rst_transitions.tab", sep="\t", quoting=3)

    # clean and augment data
    data = facts.copy(deep=True)
    data = clean(data)
    data = augment(data)
    data = data.sample(frac=1, random_state=42)

    train, test = split(data)

    # We have this many examples, stratified by number of children
    print("Train, test sizes:")
    print(train.shape, test.shape)

    oh_preprocessor = make_preprocessor(train, one_hot=True)
    ord_preprocessor = make_preprocessor(train, ordinal=True)

    encoded_oh = oh_preprocessor.fit_transform(train)
    encoded_ord = ord_preprocessor.fit_transform(train)

    # Learn
    from sklearn.svm import LinearSVC
    clf = LinearSVC()
    print("Fitting...")
    clf.fit(encoded_ord, train["label"])
    from sklearn.metrics import classification_report
    print("Fitting done. Predicting...")

    X_test = ord_preprocessor.transform(test)
    preds = clf.predict(X_test)
    print(classification_report(test["label"], preds))

    # Useful function for retrieving feature names

    names = get_column_names_from_ColumnTransformer(ord_preprocessor)
    print(names[:50])


if __name__ == '__main__':
    main()
