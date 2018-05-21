import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def _get_filter(df, model_violation, n_samples, n_feat_relevant):
    filter1 = np.ones(len(df), dtype=bool)
    if model_violation != 'all':
        filter1 = np.logical_and(
            filter1, (df.model_violation == model_violation).values)
    if n_samples != 'all':
        filter1 = np.logical_and(
            filter1, (df.n_samples == n_samples).values)
    if n_feat_relevant < 43:
        filter1 = np.logical_and(
            filter1, (df.n_feat_relevant == n_feat_relevant).values)
    else:
        print('Showing simulations for all n_feat_relevant')
    return filter1


def _plot_hexbin(x, y, n_simus):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    hb = ax.hexbin(x, y, gridsize=50,
                   norm=plt.matplotlib.colors.Normalize(0, n_simus),
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

    ax.set_xlabel(r'significance [$-log_{10}(p)$]', fontsize=20,
                  fontweight=150)
    ax.set_ylabel(r'prediction [out-of-sample $R^2$]', fontsize=20,
                  fontweight=150)
    ax.set_ylim(-0.05, 1.05)
    plt.show()


def plot_simus(df, n_simus=50, model_violation='all',
               n_samples='all',
               n_feat_relevant=43,
               pval_summary='min'):
    """Make interactive infpred plot."""
    filter1 = _get_filter(df, model_violation, n_samples, n_feat_relevant)
    pval_key = 'pvals_' + pval_summary
    pvals_, scores_ = df[[pval_key, 'scores']].values.T

    x, y = -np.log10(pvals_)[filter1], scores_[filter1]
    _plot_hexbin(x, y, n_simus)

n_samples = (list(range(50, 110, 10)) +
             list(range(200, 2100, 100)) +
             [10000, 100000, 'all'])
