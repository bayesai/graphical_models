"""
Wrapper for tabular CPD
Formats a pandas dataframe and feeds it to pgmpy.
"""
import pgmpy
import pandas as pd


class TabCPD:

    def __init__(self):
        self.evidence_features = None
        self.target = None

    def fit(self, evidence: pd.DataFrame, target: pd.Series):
        self.target = target.name if target.name is not None else "target"
        self.evidence_features = list(evidence.columns)
        evidence.loc[:, self.target] = target
        df = evidence

        # Calculating the probability of the evidence
        df_evidence = df.groupby(self.evidence_features).count()
        df_evidence = (df_evidence / len(df)).rename(columns={self.target: "pr_e"})

        # Calculating the joint probability of y and evidence
        df_joint = df.value_counts().to_frame(name="pr_joint") / len(df)

        # Calculating the conditional probability of y given evidence
        df_cpd = df_evidence.join(df_joint)
        df_cpd.loc[:, "pr_cond"] = df_cpd.loc[:, "pr_joint"] / df_cpd.loc[:, "pr_e"]

        return df_cpd

    def format_input(self, df: pd.DataFrame):
        """
        If x1 = [0, 1, 1] and x2 = [2, 2, 3], then we don't have
        (0, 3) as a possible combination, when we do need it with a probability
        of 0 (or close to 0).

        :param df:
        :return:
        """

        Pclass = list(df.Pclass.unique())
        SibSp = list(df.SibSp.unique())

        index = pd.MultiIndex.from_product([Pclass, SibSp], names = ["Pclass", "SibSp"])

        df_cond_init = pd.DataFrame(index = index).reset_index()