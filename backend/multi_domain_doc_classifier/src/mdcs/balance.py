"""
Intelligent class balancing.

Why: highly skewed classes bias generative models like MultinomialNB toward majority
priors; balanced training data yields decision boundaries that respect minority
semantics. We cap majority classes, then oversample small domains (Technology, Sports)
to the same target using bootstrap (RandomOverSampler) — a hybrid approach that
preserves real Finance/Medical diversity while giving rare domains enough weight.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler


def hybrid_balance(
    df: pd.DataFrame,
    *,
    max_per_class: int = 8_000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """
    1) Undersample any class above max_per_class.
    2) Oversample minority classes so every label reaches the same target count (capped).
    """
    work = df.copy()
    vc = work["label"].value_counts()
    # Step 1: cap
    pieces = []
    for lab, n in vc.items():
        sub = work[work["label"] == lab]
        if n > max_per_class:
            sub = sub.sample(max_per_class, random_state=random_state)
        pieces.append(sub)
    capped = pd.concat(pieces, ignore_index=True)
    vc2 = capped["label"].value_counts()
    # After capping, oversample every class to the same cap (hybrid).
    target = int(min(max_per_class, int(vc2.max())))

    idx = np.arange(len(capped)).reshape(-1, 1)
    y = capped["label"].values
    strategy = {lab: target for lab in vc2.index}
    ros = RandomOverSampler(sampling_strategy=strategy, random_state=random_state)
    idx_r, y_r = ros.fit_resample(idx, y)
    balanced = capped.iloc[idx_r.ravel()].reset_index(drop=True)

    meta = {
        "before_counts": vc.to_dict(),
        "after_cap_counts": vc2.to_dict(),
        "strategy": strategy,
        "final_counts": balanced["label"].value_counts().to_dict(),
        "method": "undersample_cap_then_random_oversample",
        "rationale": (
            "Finance/Medical dominate raw counts; Tech/Sports are tiny. Capping majority "
            "limits duplicated mass-market n-grams; oversampling minorities prevents NB "
            "class priors from washing out Technology/Sports signals."
        ),
    }
    return balanced, meta
