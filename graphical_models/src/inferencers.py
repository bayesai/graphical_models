def naive(evidence, target, df_mask):
    target_name = target.name if target.name is not None else "target"
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

    return df_mask.join(df_cpd).fillna(0)
