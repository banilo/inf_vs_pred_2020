# Authors: Denis Engemann <denis.engemann@gmail.com>
#          Danilo Bzdok <danilobzdok@gmail.com>
#
# License: BSD (3-clause)

from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as mpatches
from sklearn import metrics

df = pd.read_pickle('./simulations_may.gzip')

df.model_violation[df.model_violation.isnull()] = 'None'

df.corr_strength[df.corr_strength.isnull()] = 0

n_samples = df.n_samples.unique()

df_recon = pd.read_csv('./simultations_reconstruction_normalized.csv')

df = df.join(df_recon, lsuffix='-')

sns.set_style('ticks')

idxs = [
    df.query(
        "model_violation == 'None' & noise == 0  & corr_strength < 0.9").index,
    df.query(
        "model_violation != 'None' & noise != 0  & corr_strength == 0.9").index
]

titles = ("no model violations, 0 noise, no collinearity",
          "with model violations, noise, collinearity")


def plot_hexbin(this_df, cmap, ax, x, y, extent, title=None):

    hb = ax.hexbin(x, y, gridsize=30,
                   # norm=plt.matplotlib.colors.Normalize(0, 50),
                   norm=plt.matplotlib.colors.LogNorm(1, 300),
                   bins='log',
                   # vmin=10,
                   mincnt=1,
                   extent=extent,
                   cmap=cmap)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3%', pad=0.05)

    cb = fig.colorbar(hb, cax=cax, orientation='vertical')
    cb.set_alpha(1)
    cb.draw_all()

    cb.set_label('# simulations ', fontsize=14, fontweight=100)

    ax.set_xlabel(r"Recovery OLS", fontsize=14, fontweight=150)
    ax.set_ylabel(r"Recovery Lasso", fontsize=14, fontweight=150)
    ax.set_facecolor("dimgrey")
    ax.set_title(title)

    return


plt.close('all')
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes = axes.ravel()

for ii, ax in enumerate(axes):
    rng = np.random.RandomState(42)
    this_df = df.loc[idxs[ii]]
    x = this_df['ols_true_uncorr'] / 40
    y = this_df['lasso_true'] / 40

    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))

    plot_hexbin(ax=ax, this_df=this_df, title=titles[ii],
                cmap='plasma',
                extent=(0, 1, 0, 1),
                x=x, y=y)

plt.tight_layout()
axes[0].annotate(
    'A', xy=(-0.15, 0.98), fontweight=200, fontsize=28,
    xycoords='axes fraction')
axes[1].annotate(
    'B', xy=(-0.15, 0.98), fontweight=200, fontsize=28,
    xycoords='axes fraction')
plt.subplots_adjust(top=0.955)
plt.savefig("./figures/recovery-overview-norm-hex.png", dpi=300)
plt.savefig("./figures/recovery-overview-norm-hex.pdf")

####################################################################

cases = ("tn", "fp", "fn", "tp")
case_map = {"tn": "true negative", "fp": "false positive",
            "fn": "false negative", "tp": "true positive"}

rng = np.random.RandomState(42)

plt.close("all")

fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True, sharey=True)
axes = axes.ravel()

fig.suptitle(titles[0], fontsize=16, fontweight=150)

idx = df.query(
    "model_violation == 'None' & noise == 0  & corr_strength == 0").index

for ii, case in enumerate(cases):
    ax = axes[ii]
    this_df = df.loc[idx]
    x = this_df[f'{case}_inf_uncor'].copy()
    y = this_df[f'{case}_pred'].copy()
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
 
    plot_hexbin(ax=ax, this_df=this_df,
                title=case_map[case].capitalize(),
                extent=(0, 1, 0, 1),
                cmap='plasma',
                x=x, y=y)
    ax.set_xlabel(r"Proportion OLS", fontsize=14, fontweight=150)
    ax.set_ylabel(r"Proportion Lasso", fontsize=14, fontweight=150)

axes[0].annotate(
    'A', xy=(-0.2, 0.98), fontweight=200, fontsize=28,
    xycoords='axes fraction')
axes[1].annotate(
    'B', xy=(-0.2, 0.98), fontweight=200, fontsize=28,
    xycoords='axes fraction')
axes[2].annotate(
    'C', xy=(-0.2, 0.98), fontweight=200, fontsize=28,
    xycoords='axes fraction')
axes[3].annotate(
    'D', xy=(-0.2, 0.98), fontweight=200, fontsize=28,
    xycoords='axes fraction')

plt.tight_layout()
plt.subplots_adjust(top=0.892, bottom=0.068)

plt.savefig("./figures/pos-neg-overview-normal-norm-hex.png", dpi=300)
plt.savefig("./figures/pos-neg-overview-normal-norm-hex.pdf")

####################################################################

plt.close("all")

fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True, sharey=True)
axes = axes.ravel()
fig.suptitle(titles[1], fontsize=16, fontweight=150)

idx = df.query(
    "model_violation != 'None' & noise != 0  & corr_strength > 0").index

xys = [(0.1, 0.8), (0.7, 0.1), (0.1, 0.8), (0.7, 0.1)]
for ii, case in enumerate(cases):
    ax = axes[ii]
    this_df = df.loc[idx]
    x = this_df[f'{case}_inf_uncor'].copy()
    y = this_df[f'{case}_pred'].copy()
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))

    plot_hexbin(ax=ax, this_df=this_df,
                title=case_map[case].capitalize(),
                cmap='plasma',
                extent=(0, 1, 0, 1),
                x=x, y=y)
    ax.set_xlabel(r"Proportion OLS", fontsize=14, fontweight=150)
    ax.set_ylabel(r"Proportion Lasso", fontsize=14, fontweight=150)

axes[0].annotate(
    'A', xy=(-0.2, 0.98), fontweight=200, fontsize=28,
    xycoords='axes fraction')
axes[1].annotate(
    'B', xy=(-0.2, 0.98), fontweight=200, fontsize=28,
    xycoords='axes fraction')
axes[2].annotate(
    'C', xy=(-0.2, 0.98), fontweight=200, fontsize=28,
    xycoords='axes fraction')
axes[3].annotate(
    'D', xy=(-0.2, 0.98), fontweight=200, fontsize=28,
    xycoords='axes fraction')

plt.tight_layout()
plt.subplots_adjust(top=0.892, bottom=0.068)
plt.savefig("./figures/pos-neg-overview-path-norm-hex.png", dpi=300)
plt.savefig("./figures/pos-neg-overview-path-norm-hex.pdf")
