# Authors: Denis Engemann <denis.engemann@gmail.com>
#          Danilo Bzdok <danilobzdok@gmail.com>
#
# License: BSD (3-clause)

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

df = pd.read_pickle('./simulations_may.gzip')
df_recon = pd.read_csv('./simultations_reconstruction.csv')

df = df.join(df_recon, lsuffix='-')

factors = ("n_corr_feat", "corr_strength", "n_feat_relevant", "n_samples",
           "noise", "model_violation")

factor_labels = (
    "#corr variables",
    "collinearity",
    "#relevant variables",
    "#data points",
    "noise",
    "model violation"
)

df['n_corr_feat'][df['n_corr_feat'].isnull()] = 0
df['corr_strength'][df['corr_strength'].isnull()] = 0
df['model_violation'][df['model_violation'].isnull()] = 'none'

enc = LabelEncoder()
X = df[list(factors)].values.copy()
X[:, factors.index('model_violation')] = enc.fit_transform(
    df['model_violation'])

targets = ("inf_uncor", "pred")
cases = ("tn", "fp", "fn", "tp")
case_map = {"tn": "true negative", "fp": "false positive", "fn": "false negative",
            "tp": "true positive", "acc": "accuracy"}

params = dict(
    oob_score=True,
    n_jobs=4,
    n_estimators=1000, max_depth=len(factors))

rfs = dict()
for case in cases:
    for target in targets:
        rf = RandomForestRegressor(**params)
        rf.fit(X, df[f"{case}_{target}"])
        rfs[(case, target)] = rf

rfs[("acc", "inf_uncor")] = RandomForestRegressor(**params).fit(
    X, df["ols_true_uncorr"])
rfs[("acc", "pred")] = RandomForestRegressor(**params).fit(
    X, df["lasso_true"])

plt.close("all")
fig, axes = plt.subplots(1, 2, figsize=(9, 5), sharex=True, sharey=True)
axes = axes.ravel()
y_ax = list(range(1, len(factors) + 1))
for ii, target in enumerate(targets):
    ax = axes[ii]
    ax.plot(rfs[("acc", target)].feature_importances_ * 1e2, y_ax,
            label="accuracy",
            marker='o', mfc='white', color="black", linestyle="--")
    for case in cases:
        this_rf = rfs[(case, target)]
        ax.plot(this_rf.feature_importances_ * 1e2, y_ax, label=case_map[case],
                marker='o', mfc='white')
        ax.set_xticks(y_ax)
        ax.set_xticklabels(factors)
    ax.set_title(target)
    ax.axvline(1 / len(factors) * 1e2, color='gray', linestyle=':',
               label='unif. imp.')
    ax.set_xlabel(r"Variable Importance [%]")
axes[1].legend()
plt.tight_layout()

target_map = dict(
    pred="Lasso",
    inf_uncor="OLS"
)

color_cats = {
  "black": "#242424",
  "orange": "#EFA435",
  "sky blue": "#3EB6E7",
  "blueish green": "#009D79",
  "yellow": "#F2E55C",
  "blue": "#0076B2",
  "vermillon": "#E36C2F",
  "violet": "#D683AB"
}


plt.close("all")
fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True, sharey=False)
axes = axes.ravel()
y_ax = np.array(range(1, len(factors) + 1))

for ii, case in enumerate(cases):
    for tt, target in enumerate(targets):
        if target == "pred":
            col = color_cats["sky blue"]
        else:
            col = color_cats["vermillon"]
        ax = axes[ii]
        this_rf = rfs[(case, target)]
        # err = sem([x.feature_importances_ for x in this_rf.estimators_],
        #           axis=0)
        ax.errorbar(
            y_ax + (tt * 0.1),
            this_rf.feature_importances_ * 1e2,
            color=col,
            linewidth=2.3,
            markersize=10,
            mew=2.2,
            mfc="white",
            label=r"%s ($\mathbf{R^2 = %0.2f}$)" % (
                target_map[target], this_rf.oob_score_),
            marker='o')
        ax.set_xticks(y_ax)
        ax.set_xticklabels(factor_labels, rotation=45,
                           fontsize=11,
                           ha='right',
                           fontweight=150)
        # ax.set_title(target)
        if ii in (0, 2):
            ax.set_ylabel(r"Variable Importance [%]",
                          fontsize=12,
                          fontweight=150)
        ax.set_title(case_map[case].capitalize(), fontsize=16,
                     fontweight=150)
        ax.legend(loc="upper right")
        sns.despine(trim=True, ax=ax)

axes[0].annotate(
    'A', xy=(-0.0, 0.97), fontweight=200, fontsize=28,
    xycoords='axes fraction')
axes[1].annotate(
    'B', xy=(-0.0, 0.97), fontweight=200, fontsize=28,
    xycoords='axes fraction')
axes[2].annotate(
    'C', xy=(-0.0, 0.97), fontweight=200, fontsize=28,
    xycoords='axes fraction')
axes[3].annotate(
    'D', xy=(-0.0, 0.97), fontweight=200, fontsize=28,
    xycoords='axes fraction')

plt.tight_layout()
plt.savefig("./figures/rf-importance.png", dpi=300)
plt.savefig("./figures/rf-importance.pdf")
