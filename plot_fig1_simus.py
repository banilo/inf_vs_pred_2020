# Authors: Denis Engemann <denis.engemann@gmail.com>
#          Danilo Bzdok <danilobzdok@gmail.com>
#
# License: BSD (3-clause)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable


df = pd.read_pickle('./simulations_may.gzip')
df = df.iloc[1:]  # XXX fix with latest version

df_ols = pd.read_pickle('./simulations_may_ols.gzip')

df.model_violation[df.model_violation.isnull()] = 'None'

assert np.allclose([x[0] for x in df.lr_pvalues.values],
                   [x[0] for x in df_ols.lr_pvalues.values])

pvals = np.array([x for x in df['lr_pvalues'].values])
# catch p-values that we clipped to eps where statsmodels returned zeros
# and set them to np.finfo(np.float64).tiny.
pvals[pvals == np.finfo(pvals.dtype).eps] = 2.2250738585072014e-308
# arbitrarily filter at 1 x 10 ^ -300
pvals[pvals < 1e-300] = np.nan
scores = np.array([x for x in df['scores_debiased'].values])

scores_ = scores.max(-1)
scores_[scores_ < 0] = 0

pvals_ = pvals.min(1)

es1 = df_ols['res_rsquared']

sns.set_style('ticks')

plt.close('all')

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
ax = axes[0]
x, y = -np.log10(pvals_), scores_
color = df.n_feat_relevant.values[:]

# arbitrary size mapping
size = (np.log10(df.n_samples.values) ** 3) * 2

hb = ax.hexbin(x, y, gridsize=50,
               norm=plt.matplotlib.colors.Normalize(0, 50),
               # norm=plt.matplotlib.colors.LogNorm(),
               # bins='log',
               # vmin=10,
               mincnt=1,
               cmap='viridis')

ax.grid(True)
ax.axvline(-np.log10(0.05), color='red', linestyle='--')

divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='3%', pad=0.05)

cb = fig.colorbar(hb, cax=cax, orientation='vertical')
cb.set_alpha(1)
cb.draw_all()

cb.set_label('# simulations ', fontsize=14, fontweight=100)

sns.despine(trim=True, ax=ax)
ax.set_xlabel(r'significance [$-log_{10}(p)$]', fontsize=20, fontweight=150)
ax.set_ylabel(r'prediction [out-of-sample $R^2$]', fontsize=20, fontweight=150)
ax.set_ylim(-0.05, 1.05)
ax_zoom = axes[1]
ax_zoom.grid(True)

scat = ax_zoom.scatter(
    x, y, c='black', s=size,
    edgecolors='face',
    cmap=plt.get_cmap('viridis', len(np.unique(color))),
    vmin=0.2, vmax=color.max(),
    rasterized=True,
    alpha=0.2)

ax_zoom.set_xlim(*-np.log10([0.4, 0.02]))
ax_zoom.set_xticks(
    -np.log10([0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.01, 0.001]))
ax_zoom.set_ylim(-0.05, 1.05)
ax_zoom.axvline(-np.log10(0.05), color='red', linestyle='--')

pvals_orig = (10 ** (-1 * ax_zoom.get_xticks())).round(3)
ax_zoom.set_xticklabels(pvals_orig)
ax_zoom.set_xlabel(r'significance [$p$-value]', fontsize=20, fontweight=150)
ax_zoom.set_ylabel(
    r'prediction [out-of-sample $R^2$]', fontsize=20, fontweight=150)

ax_es = axes[2]
ax_es.grid(True)

ax.annotate(
    'A', xy=(-0.17, 0.95), fontweight=200, fontsize=30,
    xycoords='axes fraction')
ax_zoom.annotate(
    'B', xy=(-0.17, 0.95), fontweight=200, fontsize=30,
    xycoords='axes fraction')
ax_es.annotate(
    'C', xy=(-0.17, 0.95), fontweight=200, fontsize=30,
    xycoords='axes fraction')

sns.despine(trim=True, ax=ax_zoom)

es1 = df_ols['res_rsquared']

hb = ax_es.hexbin(es1, y, gridsize=50,
                  norm=plt.matplotlib.colors.Normalize(0, 50),
                  # norm=plt.matplotlib.colors.LogNorm(),
                  # bins='log',
                  vmin=1,
                  mincnt=1,
                  cmap='viridis')
ax_es.grid(True)

divider = make_axes_locatable(ax_es)
cax = divider.append_axes('right', size='3%', pad=0.05)

cb = fig.colorbar(hb, cax=cax, orientation='vertical')
cb.set_alpha(1)
cb.draw_all()

cb.set_label('# simulations ', fontsize=14, fontweight=100)
sns.despine(trim=True, ax=ax_es)

ax_es.set_xlabel(r'effect size [in-sample $R^2$]', fontsize=20,
                 fontweight=150)
ax_es.set_ylabel(r'prediction [out-of-sample $R^2$]', fontsize=20,
                 fontweight=150)
ax_es.set_ylim(-0.05, 1.05)

plt.subplots_adjust(left=0.05, right=0.96, top=0.95, bottom=0.14, wspace=0.37)

fig.savefig('./figures/simulations_overview_fig1.png', bbox_inches='tight',
            dpi=300)
fig.savefig('./figures/simulations_overview_fig1.pdf', bbox_inches='tight',
            dpi=300)