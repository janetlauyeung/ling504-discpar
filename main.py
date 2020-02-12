from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC
from clean import clean
from augment import augment
from trial import Trial

################################################################################
# Begin global state
################################################################################

facts = pd.read_csv("rst_transitions.tab", sep="\t", quoting=3)

# some categories of columns
lex_feats = ["Top2-Stack", "Top1Span", "First-Queue"]
categorical_features = ['Top12-StackXML', 'Stack-QueueSType', "Stack", "genre", "Stack-QueueSameSent",
                        "Top12-StackSameSent",
                        'Top12-StackSameSent', 'Stack-QueueXML', 'Top12-StackSType',
                        "Top12-StackDir", "Stack-QueueDir", "First-QueueEduFunc", "Top1SpanEduFunc"]
numeric_features = ['First-QueueDist-To-Begin', 'Top2-StackLength-EDU', 'Top1-StackLength-EDU'] #'Top1-StacknEDUs']
scale_features = ['Top2-StackDist-To-End', 'First-Queue-Len']
text_features = ['First-Queue', 'Top1Span', 'Top2-Stack']

# clean and augment data
data = facts.copy(deep=True)
data = clean(data)
data = augment(data)
data = data.sample(frac=1, random_state=42)
def split(data):
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in splitter.split(data, data["label"]):
        train = data.loc[train_idx]
        test = data.loc[test_idx]
    return train, test
train, test = split(data)

################################################################################
# Feature engineering
################################################################################
# This is effectively a 'do nothing transformer', we may need it below
nop_transformer = FunctionTransformer(lambda x: x)

def make_transformer(data, one_hot=False, ordinal=False):
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
        ('identity', nop_transformer)
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

################################################################################
# Trials
################################################################################

# A trial is an object that conceptually means "a model run with a featureset"
# You hand it a ColumnTransformer in its constructor, and in return, it will:
# - evaluate on test for you
# - store the model in trial.model
# - store the preds in trial.preds
# - store the transformer in trial.transformer
# and more!

class XGBTrial(Trial):
    def __init__(self, transformer, use_test=False, **kwargs):
        self.method = "decision_function"
        super().__init__(**kwargs)

        eval_rows = test if use_test else train

        X = transformer.fit_transform(train)
        y = train["label"]

        model = XGBClassifier(
            nthread=-1
        )
        model.fit(X, y)

        # predict
        X_eval = transformer.transform(eval_rows)
        preds = model.predict(X_eval)

        # hold on to refs in case we want them later
        self.X = X
        self.y = y
        self.model = model
        self.preds = preds
        self.transformer = transformer

        # populate score attributes
        self._perf(eval_rows["label"], preds)

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


if __name__ == '__main__':
    # We have this many examples, stratified by number of children
    print("Note: run this file with -i (e.g.: `python -i main.py`) to drop into a Python REPL")
    print("Train, test sizes:")
    print(train.shape, test.shape)

    oh_transformer = make_transformer(train, one_hot=True)
    ord_transformer = make_transformer(train, ordinal=True)

    print("Beginning XGB fitting...")
    xgb_trial = XGBTrial(ord_transformer, use_test=True)
    print(xgb_trial)
    print("Fitting done. Predicting...")

    print(classification_report(test["label"], xgb_trial.preds))

    names = get_column_names_from_ColumnTransformer(ord_transformer)
    print(names[:50])
