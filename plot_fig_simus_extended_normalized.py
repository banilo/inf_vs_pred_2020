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


def plot_core(this_df, cmap, ax, x, y, title=None, corr=False):
    m1 = x > y
    m2 = x == y
    m3 = x < y
    dd = np.mean(y - x)
    sr = stats.spearmanr(x, y)

    x_ = x + rng.rand(len(x)) * 0.01
    y_ = y + rng.rand(len(y)) * 0.01

    if title is None:
        title = ""
    if corr:
        title += r'$r_{sp}$' + rf'$={sr[0]:0.2f}$'
        title += r', $M(Lasso-OLS)$' + rf'$={dd:0.1f}$'

    ax.set_title(title, fontsize=14, fontweight=150, pad=18)

    z = (this_df['n_samples'])
    markers = (">", "o", "^")
    for jj, mask in enumerate((m1, m2, m3)):
        if np.any(mask):
            scat0 = ax.scatter(np.arange(1, 41)[4::5] + 1000,
                       np.arange(1, 41)[4::5] + 1000,
                       s=np.arange(1, 41)[4::5],
                       alpha=0.2,
                       edgecolor='face',
                       marker=markers[jj],
                       norm=matplotlib.colors.Normalize()
                       )
            ax.set_xlim(-0.03, 1.03)
            ax.set_ylim(-0.03, 1.03)
            legend = ax.legend(
                # title='#rel. variables',
                *scat0.legend_elements(
                    "sizes",
                    num=list(range(1, 41))[4::5],
                    color='black'),
                ncol=8,
                shadow=False,
                fancybox=False,
                frameon=False,
                bbox_to_anchor=(0, 1),
                loc="lower left",
                prop={'size': 5})

            s = this_df['n_feat_relevant'].loc[mask]
            scat = ax.scatter(x_[mask], y_[mask], c=z[mask], cmap=cmap,
                              s=s,
                              alpha=0.2,
                              edgecolor='face',
                              marker=markers[jj],
                              rasterized=True,
                              norm=matplotlib.colors.LogNorm(
                                vmin=50, vmax=100000))
            

    frame = legend.get_frame()
    # frame.set_linewidth(0.3)
    # frame.set_edgecolor('white')
    # frame.set_facecolor('black')
    # plt.setp(legend.get_texts(), color='w')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3%', pad=0.05)
    cb = fig.colorbar(scat, cax=cax, orientation='vertical')
    cb.set_alpha(1)
    cb.draw_all()
    cb.set_label('n samples ', fontsize=14, fontweight=100)

    ax.set_xlabel(r"Recovery OLS", fontsize=14, fontweight=150)
    ax.set_ylabel(r"Recovery Lasso", fontsize=14, fontweight=150)
    ax.set_facecolor("black")

    return sr, dd


plt.close('all')
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes = axes.ravel()

for ii, ax in enumerate(axes):
    rng = np.random.RandomState(42)
    this_df = df.loc[idxs[ii]]
    x = this_df['ols_true_uncorr'] / 40
    y = this_df['lasso_true'] / 40

    # hb = ax.hexbin(x, y, gridsize=30,
    #                # norm=plt.matplotlib.colors.LogNorm(0, 50),
    #                norm=plt.matplotlib.colors.LogNorm(),
    #                # bins='log',
    #                # vmin=10,
    #                mincnt=1,
    #                cmap='plasma')

    sr, dd = plot_core(ax=ax, this_df=this_df, title=titles[ii],
                       cmap='plasma',
                       x=x, y=y, corr=False)
    ax.set_facecolor("dimgrey")
    annot = r'$r_{sp}$' + rf'$={sr[0]:0.2f}$'
    annot += '\n' + r'$M_{(Lasso-OLS)}$' + rf'$={dd:0.1f}$'
    ax.annotate(
        annot, xy=(0.6, 0.05), xycoords='axes fraction',
        color='white')
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))

plt.tight_layout()
axes[0].annotate(
    'A', xy=(-0.15, 0.98), fontweight=200, fontsize=28,
    xycoords='axes fraction')
axes[1].annotate(
    'B', xy=(-0.15, 0.98), fontweight=200, fontsize=28,
    xycoords='axes fraction')

plt.savefig("./figures/recovery-overview-norm.png", dpi=300)
plt.savefig("./figures/recovery-overview-norm.pdf")

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
    sr, dd = plot_core(ax=ax, this_df=this_df,
                       title=case_map[case].capitalize(),
                       cmap='plasma',
                       x=x, y=y, corr=False)
    ax.set_facecolor("dimgrey")
    annot = ''
    if not np.isnan(sr[0]):
        annot += r'$r_{sp}$' + rf'$={sr[0]:0.2f}$'
    annot += '\n' + r'$M_{(Lasso-OLS)}$' + rf'$={dd:0.1f}$'
    ax.annotate(
        annot, xy=(0.5, 0.1), xycoords='axes fraction',
        color='white')
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
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
plt.savefig("./figures/pos-neg-overview-normal-norm.png", dpi=300)
plt.savefig("./figures/pos-neg-overview-normal-norm.pdf")

####################################################################

plt.close("all")

fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True, sharey=True)
axes = axes.ravel()
fig.suptitle(titles[1], fontsize=16, fontweight=150)

idx = df.query(
    "model_violation != 'None' & noise != 0  & corr_strength > 0").index

xys = [(0.05, 0.8), (0.6, 0.1), (0.05, 0.8), (0.6, 0.1)]
for ii, case in enumerate(cases):
    ax = axes[ii]
    this_df = df.loc[idx]
    x = this_df[f'{case}_inf_uncor'].copy()
    y = this_df[f'{case}_pred'].copy()
    sr, dd = plot_core(
        ax=ax, this_df=this_df, title=case_map[case].capitalize(),
        cmap='plasma',
        x=x, y=y, corr=False)
    ax.set_facecolor("dimgrey")
    annot = r'$r_{sp}$' + rf'$={sr[0]:0.2f}$'
    annot += '\n' + r'$M_{(Lasso-OLS)}$' + rf'$={dd:0.1f}$'
    ax.annotate(
        annot, xy=xys[ii], xycoords='axes fraction',
        color='white')
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
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
plt.savefig("./figures/pos-neg-overview-path-norm.png", dpi=300)
plt.savefig("./figures/pos-neg-overview-path-norm.pdf")
