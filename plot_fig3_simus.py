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
# catch p-values that we clipped to eps where statsmodels returned zeros
# and set them to np.finfo(np.float64).tiny.
pvals[pvals == np.finfo(pvals.dtype).eps] = 2.2250738585072014e-308
# arbitrarily filter at 1 x 10 ^ -300
pvals[pvals < 1e-300] = np.nan

scores = np.array([x for x in df['scores_debiased'].values])

scores_ = scores.max(-1)
scores_[scores_ < 0] = 0

pvals_ = pvals.min(1)

sns.set_style('ticks')

plt.close('all')

titles = [
    'Sample size',
    'Proportion relevant',
    'Multi-collinearity',
    'Corruption with noise',
    'True model or not',
    'Polynomial degree',
]

cases = [
    'sample size',
    'prop rel',
    'correlation',
    'noise',
    'model',
    'poly'
]

data_index = {
    'model': np.arange(len(df)),
    'poly': np.arange(len(df)),
    'noise': np.arange(len(df)),
    'correlation': np.arange(len(df)),
    'sample size': np.arange(len(df)),
    'prop rel': np.arange(len(df)),
}

sample_ticks = [50, '', '', '', '', 100, '', '', '', 500, '', '', 800, '',
                1000, '', 1200, '', 1400, 1500, 1600, 1700, 1800, 1900,
                2000, 10000, 100000]

plt.close('all')
fig, axes = plt.subplots(2, 3, figsize=(9, 5), sharey=True)

n_samp = df.n_samples
axes = axes.ravel()
for i_case, case in enumerate(cases):

    ax = axes[i_case]
    inds = data_index[case]
    x, y = -np.log10(pvals_[inds]), scores_[inds]

    color = df.n_feat_relevant.values[inds]

    ax.grid(True)
    if case == 'model':
        cvals = (df.model_violation != 'None').values.astype(int)
    elif case == 'poly':
        cvals = np.array([
            int(x.split('^')[-1]) if '^' in x else 0 for x in
            df.model_violation.values[inds]])
    elif case == 'noise':
        cvals = df.noise.values
    elif case == 'correlation':
        cvals = np.zeros(len(df))
        cvals[df.corr_strength == .5] = 0.5
        cvals[df.corr_strength == .9] = 0.9
    elif case == 'sample size':
        cvals = df.n_samples.values
    elif case == 'prop rel':
        cvals = (df.n_feat_relevant / 40.).values
    else:
        raise ValueError('No no no.')

    if case == 'sample size':
        ordered_range = np.logspace(0, 1, len(np.unique(cvals))) / 10.
    else:
        ordered_range = np.linspace(0, 1, len(np.unique(cvals)))
    unique_vals = sorted(np.unique(cvals))
    assert len(ordered_range) == len(unique_vals)
    cvals_plt = pd.Series(cvals).map(
        dict(zip(unique_vals, ordered_range))).values

    print(case)
    # print(ordered_range)
    print(np.unique(cvals_plt))

    scat = ax.scatter(x, y, c=cvals_plt,
                      s=1,
                      edgecolors='face',
                      # cmap='viridis',
                      cmap=plt.get_cmap('viridis', len(np.unique(cvals))),
                      # norm=plt.matplotlib.colors.BoundaryNorm(
                      #     boundaries=np.unique(cvals),
                      #     ncolors=len(np.unique(cvals))),
                      rasterized=True,
                      alpha=0.5)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.3)

    cticks = np.unique(cvals)
    cb = fig.colorbar(scat, cax=cax, orientation='vertical',
                      ticks=np.unique(cvals_plt))
    cb.set_alpha(1)
    cb.draw_all()
    cax.yaxis.set_label_position('left')
    # if case == 'sample size':
    cb.set_ticklabels(
        unique_vals if case != 'sample size' else
        sample_ticks)
    if case == 'model':
        cb.set_ticklabels(['no', 'yes'])

    cax.tick_params(labelsize=8)

    ax.set_xlim(-5, 305)
    ax.set_xticks([0, 100, 200, 300])

    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([
        0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 0.8, 1.0])
    ax.set_title(titles[i_case], fontsize=10)
    ax.axvline(-np.log10(0.05), color='red', linestyle='--')

    ax_inset = inset_axes(
        ax, width="30%",
        height="30%",
        loc=4,
        bbox_to_anchor=(0, 0.1, 1, 1.1),
        bbox_transform=ax.transAxes)
    scat = ax_inset.scatter(x, y,
                            c=cvals,
                            cmap=plt.get_cmap('viridis',
                                              len(np.unique(cvals_plt))),
                            s=1,
                            edgecolors='face',
                            vmin=0.2,
                            rasterized=True,
                            alpha=0.5)
    ax_inset.set_xlim(*-np.log10([0.5, 0.001]))
    ax_inset.set_xticks(
        -np.log10([0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.001]))
    ax_inset.set_xticklabels(
        ['', '', '', '', '', 0.05, '', 0.001],
        color='#383838')

    ax_inset.set_ylim(-0.05, 1.05)
    ax_inset.axvline(
        -np.log10(0.05), color='red', linestyle='--')
    pvals_orig = (10 ** (-1 * ax_inset.get_xticks())).round(3)
    ax_inset.set_yticklabels(['', 0, 1, ''], color='#383838')
    sns.despine(trim=True, ax=ax)
    ax_inset.tick_params(labelsize=8)
    if i_case > 2:
        ax.set_xlabel(r'$-log_{10}(p)$', fontsize=10, fontweight=150)
    if i_case in (0, 3):
        ax.set_ylabel(r'prediction [$R^2$]',
                      fontsize=12, fontweight=150)
    ax.annotate(
        'ABCDEFG'[i_case], xy=(-0.20, 0.99), fontweight=200, fontsize=20,
        xycoords='axes fraction')


plt.subplots_adjust(hspace=0.33, wspace=.46, left=.07, right=.94, top=.94,
                    bottom=.10)
plt.savefig('./figures/simulations_by_aspect.png', bbox_inches='tight',
            dpi=300)
plt.savefig('./figures/simulations_by_aspect.pdf', bbox_inches='tight',
            dpi=300)
