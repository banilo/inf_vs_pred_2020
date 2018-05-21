# Authors: Denis Engemann <denis.engemann@gmail.com>
#          Danilo Bzdok <danilobzdok@gmail.com>
#
# License: BSD (3-clause)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable

df = pd.read_pickle('./simulations_may.gzip')

df.model_violation[df.model_violation.isnull()] = 'None'


pvals = np.array([x for x in df['lr_pvalues'].values])
pvals[pvals == np.finfo(pvals.dtype).eps] = 2.2250738585072014e-308
pvals[pvals < 1e-300] = np.nan

scores = np.array([x for x in df['scores_debiased'].values])

scores_ = scores.max(-1)
scores_[scores_ < 0] = 0

pvals_ = pvals.min(1)

sns.set_style('ticks')

plt.close('all')


pathologies = [
    'None', 'abs', 'log', 'exp', 'sqrt', '1/x', 'x^2', 'x^3', 'x^4',
    'x^5']

pathologies_map = [
    'True model', 'abs', 'log', 'exp', 'sqrt',
    r'$\frac{1}{x}$', r'$x^{2}$', r'$x^{3}$', r'$x^{4}$',
    r'$x^{5}$']

plt.close('all')
fig, axes = plt.subplots(2, 5, figsize=(11, 5), sharey=True)

n_samp = df.n_samples
axes = axes.ravel()
for ii, path in enumerate(pathologies):
    ax = axes[ii]
    inds = np.where(df.model_violation == path)
    x, y = -np.log10(pvals_[inds]), scores_[inds]

    color = df.n_feat_relevant.values[inds]
    size = (np.log10(df.n_samples.values) ** 3) * 3

    ax.grid(True)
    hb = ax.hexbin(x, y, gridsize=20,
                   norm=plt.matplotlib.colors.Normalize(0, 50),
                   mincnt=1,
                   cmap='viridis')
    if ii in [4, 9]:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.1)

        cb = fig.colorbar(hb, cax=cax, orientation='vertical')
        cb.set_label('# simulations ', fontsize=14, fontweight=100)
    ax.set_xlim(-1, 305)
    ax.set_xticks([0, 100, 200, 300])
    # ax.set_facecolor(plt.matplotlib.cm.viridis(0))
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([
        0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 0.8, 1.0])
    ax.set_title(pathologies_map[ii])
    ax.axvline(-np.log10(0.05), color='red', linestyle='--')

    ax_inset = inset_axes(
        ax, width="30%",  # width = 30% of parent_bbox
        height="30%",  # height : 1 inch
        # borderpad=2,
        loc=4,
        bbox_to_anchor=(0, 0.1, 1, 1.1),
        bbox_transform=ax.transAxes)
    scat = ax_inset.scatter(x, y, c="black", s=size,
                            edgecolors='face',
                            vmin=0.2, vmax=color.max(),
                            alpha=0.1)
    ax_inset.set_xlim(*-np.log10([0.5, 0.001]))
    ax_inset.set_xticks(
        -np.log10([0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.01, 0.001]))
    ax_inset.set_ylim(-0.05, 1.05)
    ax_inset.axvline(
        -np.log10(0.05), color='red', linestyle='--')
    pvals_orig = (10 ** (-1 * ax_inset.get_xticks())).round(3)
    ax_inset.set_xticklabels([
        str(ll) if ii in [0, len(pvals_orig) - 1]
        else '' for ii, ll in enumerate(pvals_orig)], color='#383838')
    ax_inset.set_yticklabels(['', 0, 1, ''], color='#383838')
    sns.despine(trim=True, ax=ax)

    if ii > 4:
        ax.set_xlabel(r'$-log_{10}(p)$', fontsize=10, fontweight=150)
    if ii in (0, 5):
        ax.set_ylabel(r'prediction [$R^2$]',
                      fontsize=12, fontweight=150)

plt.subplots_adjust(hspace=0.33, left=.08, right=.94, top=.94, bottom=.10)
plt.savefig('./figures/simulations_by_violation.png', bbox_inches='tight',
            dpi=300)
