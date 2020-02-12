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
            self.micro_recall = recall_score(gold, pred, average='micro')
            self.macro_recall = recall_score(gold, pred, average='macro')
            self.micro_f1 = f1_score(gold, pred, average='micro')
            self.macro_f1 = f1_score(gold, pred, average='macro')

    def __repr__(self):
        s = f'{self.__class__.__name__}'
        s += f' {self.name}' if hasattr(self, 'name') else ''
        s += ':'
        s += ('\n  features: ' + ', '.join(self.features)) if hasattr(self, 'features') else ''
        s += f'\n  rmse: {self.rmse}' if hasattr(self, 'rmse') else ''
        s += f'\n  accuracy: {self.accuracy}' if hasattr(self, 'accuracy') else ''
        s += f'\n  micro_recall: {self.micro_recall}' if hasattr(self, 'micro_recall') else ''
        s += f'\n  macro_recall: {self.macro_recall}' if hasattr(self, 'macro_recall') else ''
        s += f'\n  micro_f1: {self.micro_f1}' if hasattr(self, 'micro_f1') else ''
        s += f'\n  macro_f1: {self.macro_f1}' if hasattr(self, 'macro_f1') else ''
        s += '\n'
        return s
