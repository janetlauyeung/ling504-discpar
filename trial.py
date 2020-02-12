import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score, recall_score, f1_score

class Trial:
    def __init__(self, *args, **kwargs):
        if 'name' in kwargs:
            self.name = kwargs.pop('name')

    def _perf(self, gold, pred, regression=False):
        if regression:
            self.rmse = np.sqrt(mean_squared_error(gold, pred))
        else:
            self.accuracy = accuracy_score(gold, pred)
            self.recall = recall_score(gold, pred)
            self.f1 = f1_score(gold, pred)

    def __repr__(self):
        s = f'{self.__class__.__name__}'
        s += f' {self.name}' if hasattr(self, 'name') else ''
        s += ':'
        s += ('\n  features: ' + ', '.join(self.features)) if hasattr(self, 'features') else ''
        s += f'\n  accuracy: {self.accuracy}' if hasattr(self, 'accuracy') else ''
        s += f'\n  recall: {self.recall}' if hasattr(self, 'recall') else ''
        s += f'\n  rmse: {self.rmse}' if hasattr(self, 'rmse') else ''
        s += f'\n  f1: {self.f1}' if hasattr(self, 'f1') else ''
        s += '\n'
        return s
