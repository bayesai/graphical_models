"""
Wrapper for tabular CPD
Formats a pandas dataframe and feeds it to pgmpy.
"""
import pgmpy
import pandas as pd
from . import inferencers


class TabCPD:

    def __init__(self, inferencer: str, **hyperparameters):
        self.evidence_features = None
        self.inferencer = getattr(inferencers, inferencer)
        self.target = None
        self.inferenced = None
        self.hyperparameters = hyperparameters

    def fit(self, evidence: pd.DataFrame, target: pd.Series):
        self.target = target.name if target.name is not None else "target"
        self.evidence_features = list(evidence.columns)
        df_mask = pd.DataFrame(index=pd.MultiIndex.from_product(
            [sorted(list(evidence.loc[:, col].unique())) for col in evidence.columns] + [list(target.unique())]
            , names=list(evidence.columns) + [self.target]))

        self.inferenced = self.inferencer(**self.hyperparameters)(evidence, target, df_mask)

        return self.cpd

    @property
    def cpd(self):
        return self.inferenced.cpd

    @property
    def target_cpd(self):
        return self.inferenced.target_cpd
