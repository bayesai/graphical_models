"""
Wrapper for tabular CPD
Formats a pandas dataframe and feeds it to pgmpy.
"""
import pgmpy
import pandas as pd
from . import inferencers


class TabCPD:

    def __init__(self, inferencer):
        self.evidence_features = None
        self.inferencer = getattr(inferencers, inferencer)

    def fit(self, evidence: pd.DataFrame, target: pd.Series):
        self.target = target.name if target.name is not None else "target"
        self.evidence_features = list(evidence.columns)
        df_mask = pd.DataFrame(index=pd.MultiIndex.from_product(
            [sorted(list(evidence.loc[:, col].unique())) for col in evidence.columns] + [list(target.unique())]
            , names=list(evidence.columns) + [self.target]))

        df_cpd = self.inferencer(evidence, target, df_mask)

        return df_cpd
