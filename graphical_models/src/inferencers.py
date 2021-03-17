import pandas as pd


class PriorParams:
    def __init__(self, default_value, params):
        self._params = params
        self.default_value = default_value

    def __getitem__(self, item):
        result = self._params.get(item)
        if result is None:
            result = self.default_value
        return result


class Naive:
    def __init__(self):
        self.cpd = None
        self.unique_index_vals = None
        self.target_name = None

    def __call__(self, evidence, target, df_mask):
        target_name = target.name if target.name is not None else "target"
        self.target_name = target_name
        self.unique_index_vals = sorted(target.unique())
        evidence_features = list(evidence.columns)
        evidence.loc[:, target_name] = target
        df = evidence

        # Calculating the probability of the evidence
        df_evidence = df.groupby(evidence_features).count()
        df_evidence = (df_evidence / len(df)).rename(columns={target_name: "pr_e"})

        # Calculating the joint probability of y and evidence
        df_joint = df.value_counts().to_frame(name="pr_joint") / len(df)

        # Calculating the conditional probability of y given evidence
        df_cpd = df_evidence.join(df_joint)
        df_cpd.loc[:, "pr_cond"] = df_cpd.loc[:, "pr_joint"] / df_cpd.loc[:, "pr_e"]

        self.cpd = df_mask.join(df_cpd).fillna(0)
        return self

    @property
    def target_cpd(self):
        temp = self.cpd.drop(columns=["pr_e", "pr_joint"])
        result = temp.loc[:, :, self.unique_index_vals[0]].rename(
            columns={"pr_cond": f"{self.target_name}_{self.unique_index_vals[0]}"})
        for target_val in self.unique_index_vals[1:]:
            result = result.join(temp.loc[:, :, target_val].rename(
                columns={"pr_cond": f"{self.target_name}_{target_val}"}))
        return result


class Dirichlet:
    def __init__(self, alphas):
        self.prior_params = PriorParams(default_value=1, params=alphas)
        self.cpd = None

    def __call__(self, evidence, target, df_mask):
        target_name = target.name if target.name is not None else "target"
        self.target_name = target_name
        self.unique_index_vals = sorted(target.unique())
        evidence_features = list(evidence.columns)
        evidence.loc[:, target_name] = target
        df = evidence
        total_count = len(evidence)

        # Calculating the probability of the evidence
        df_evidence = df.groupby(evidence_features).count()
        df_evidence = (df_evidence / len(df)).rename(columns={target_name: "pr_e"})

        # Calculating the joint probability of y and evidence
        df_counts = df.value_counts().to_frame(name="counts")
        df_counts = df_counts.join(df_mask).sort_index().fillna(0)
        values = list()
        for i in df_counts.iterrows():
            values.append(self.prior_params[i[0]])
        df_counts.loc[:, "prior_hyperparam"] = values
        df_counts.loc[:, "post_hyperparam"] = df_counts.loc[:, "counts"] + df_counts.loc[:, "prior_hyperparam"]
        cpd = df_counts.loc[:, "post_hyperparam"] / df_counts.loc[:, "post_hyperparam"].groupby(df_counts.index.names[:-1]).sum()
        self.cpd = cpd.to_frame(name="posterior_distribution")
        return self

    @property
    def target_cpd(self):
        temp = self.cpd.loc[:, ["posterior_distribution"]]
        result = temp.loc[:, :, self.unique_index_vals[0]].rename(
            columns={"posterior_distribution": f"{self.target_name}_{self.unique_index_vals[0]}"})
        for target_val in self.unique_index_vals[1:]:
            result = result.join(temp.loc[:, :, target_val].rename(
                columns={"posterior_distribution": f"{self.target_name}_{target_val}"}))
        return result

"""
Bayes: 
P(theta | X) = P(X | theta) * P(theta) / P(X)


(alpha 1, ..., alpha k): hyperparameters for the prior
theta ~ Dir(alpha 1, ..., alpha k) prior for theta 
X | theta ~ Cat(theta 1, ..., theta k) sampling distribution
Observed / data:
Data = (c1, ..., ck): counts that we observed (our data)
Update / inference:
theta | Data ~ Dir(alpha 1 + c1, ..., alpha k + ck)

Prediction:
P(Xj = j) = integral(   P(Xj = j | theta)  * Pt(theta)   d_theta)
P(Xj | theta) is constant.
Pt(theta) is what changes. We update this and this changes our prediction. 
"""
