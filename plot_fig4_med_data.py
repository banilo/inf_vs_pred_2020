# imports and plotting utility functions
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.cross_validation import ShuffleSplit
from sklearn.linear_model import Lasso
from matplotlib import pylab as plt
from statsmodels.regression.linear_model import OLS
from scipy.linalg import norm

import sklearn.datasets as ds



def compute_lasso_regpath(X, y, C_grid, metric=None, verbose=True):
    """Compute the lasso path."""
    coef_list2 = []
    acc_list2 = []
    acc_unbiased_list2 = []
    nonzero_list2 = []
    if metric is None:
        metric = r2_score
    for i_step, my_C in enumerate(C_grid):
        sample_accs = []
        sample_accs_unbiased = []
        sample_coef = []
        for i_subsample in range(100):
            folder = ShuffleSplit(n=len(y), n_iter=100, test_size=0.1,
                                  random_state=i_subsample)
            train_inds, test_inds = next(iter(folder))

            clf = Lasso(alpha=my_C, random_state=i_subsample)

            clf.fit(X[train_inds, :], y[train_inds])
            acc = metric(
                y_true=y[test_inds],
                y_pred=clf.predict(X[test_inds]))

            # get out-of-sample accuracy from unbiased linear model with
            # selected inputs
            b_vars_to_keep = clf.coef_ != 0
            if np.sum(b_vars_to_keep) > 0:
                unbiased_lr = LinearRegression()
                unbiased_lr.fit(
                    X[train_inds, :][:, b_vars_to_keep], y[train_inds])
                unbiased_acc = metric(
                    y_true=y[test_inds],
                    y_pred=unbiased_lr.predict(
                        X[test_inds][:, b_vars_to_keep]))
            else:
                unbiased_acc = 0

            sample_accs.append(acc)
            sample_accs_unbiased.append(unbiased_acc)
            sample_coef.append(clf.coef_)

        mean_coefs = np.mean(np.array(sample_coef), axis=0)
        coef_list2.append(mean_coefs)
        acc_for_C = np.mean(sample_accs)
        acc_for_C_unbaised = np.mean(sample_accs_unbiased)
        acc_list2.append(acc_for_C)
        acc_unbiased_list2.append(np.mean(sample_accs_unbiased))
        notzero = np.count_nonzero(mean_coefs)
        nonzero_list2.append(notzero)
        if verbose:
            print(
                "alpha: %.4f acc: %.2f / %.2f (unbiased) active_coefs: %i" % (
                    my_C, acc_for_C, acc_for_C_unbaised, notzero))
    return (np.array(coef_list2), np.array(acc_list2), np.array(nonzero_list2),
            np.array(acc_unbiased_list2))


def _get_data(case):
    if case == 'case1':
        df_birth = pd.read_csv('dataset_birthwt.csv')
        df_part1 = StandardScaler().fit_transform(df_birth[['age', 'lwt']])
        df_part2 = df_birth[['race', 'smoke', 'ptl', 'ht', 'ui', 'ftv']]
        feat_names = ['age', 'lwt'] + list(df_part2.columns)
        y = StandardScaler().fit_transform(
            df_birth['bwt'].values[:, None])[:, 0]
        X = np.hstack((df_part1, df_part2))
    elif case == 'case2':
        df_prostate = pd.read_csv('dataset_prostate.csv')
        y = df_prostate['lpsa']
        feat_names = ['lcavol', 'lweight', 'age', 'lbph', 'svi', 'lcp',
                      'gleason',
                      'pgg45']
        X = StandardScaler().fit_transform(df_prostate[feat_names])
    elif case == 'case3':
        bun = ds.load_diabetes()
        X, y = bun.data, bun.target
        X = StandardScaler().fit_transform(X)
        feat_names = bun.feature_names
    elif case == 'case4':
        df_fev = pd.read_csv('dataset_FEV.csv')

        df_fev.drop(labels='id', axis=1, inplace=True)

        feat_names = ['age', u'fev', u'height', u'sex', u'smoke']
        df_part1 = pd.DataFrame(StandardScaler().fit_transform(
            df_fev[feat_names[:-2]].values), columns=feat_names[:-2])
        df_part2 = pd.get_dummies(df_fev[feat_names[-2:]], drop_first=True)
        y = StandardScaler().fit_transform(
            df_part1['fev'].values[:, None])[:, 0]
        df_part1.drop(labels='fev', axis=1, inplace=True)
        X = np.hstack((df_part1.values, df_part2.values))
        feat_names = list(df_part1.columns) + list(df_part2.columns)
        feat_names = feat_names[:-1] + ['smoker']
    return X, y, feat_names


def _run_infpred(case):
    X, y, feat_names = _get_data(case)

    model = OLS(y, X)
    res = model.fit()
    lr_coefs = res.params
    lr_pvalues = res.pvalues

    snr = (norm(a=lr_coefs, ord=2) ** 2) / (norm(a=res.resid, ord=2) ** 2)
    print('Signal-to-noise ratio: %.4f' % snr)

    # compute Lasso regularization paths
    C_grid = np.logspace(-2, 2, 25)
    coef_list, acc_list, nonzero_list, unbiased_acc_list =\
        compute_lasso_regpath(X, y, C_grid)

    return unbiased_acc_list, lr_pvalues, coef_list, feat_names

cases = [
    ('case1', 'Birthweight',
     'Birthweight Data\nsignificant, but hard to predict'),
    ('case2', 'Prostata',
     'Prostata Data\nPredictive but not significant'),
    ('case3', 'Diabetes',
     'Diabetes Data\nPredictive and some significant'),
    ('case4', 'FEV',
     'FEV Data\nsignificant but largely\nignorable for prediction'),
]

results = [_run_infpred(case) for case, _, _ in cases]


def _plot_infpred(unbiased_acc_list, lr_pvalues, coef_list, feat_names,
                  acc_offset=0.1, annot_ha='center', ax=None):
    if ax is None:
        fig = plt.figure(figsize=(9, 9))
        ax = plt.gca()
    else:
        fig = None
    sorter = unbiased_acc_list.argsort()[::-1]
    colors = plt.cm.viridis_r(np.linspace(0.1, 0.9, len(sorter)))

    unique_nonzero = {}
    size = 20
    for ii, idx in enumerate(sorter):
        acc = unbiased_acc_list[idx]
        non_zero = np.where(coef_list[idx])[0]
        if tuple(non_zero) not in unique_nonzero:
            unique_nonzero[tuple(non_zero)] = non_zero
        else:
            print('skipping', ii)
            continue

        xx = -np.log10(lr_pvalues[non_zero])
        this_acc = np.array([acc] * len(xx))
        size *= 0.9
        ax.plot(xx + np.random.sample(len(xx)) * 0.01,
                this_acc,
                marker='o', linestyle='None',
                color=colors[ii], zorder=-ii,
                alpha=0.9,
                mfc='None',
                mew=1,
                markersize=size)
        if ii == 0:
            psorter = np.argsort(lr_pvalues)
            feat_names_ = [feat_names[kk] for kk in psorter]
            xx2 = -np.log10(lr_pvalues[psorter])
            for jj, (this_name, this_x) in enumerate(zip(feat_names_, xx2)):
                print(this_x)
                ax.annotate(
                    this_name, xy=(
                        this_x, acc + acc_offset + (0.1 if jj % 2 else 0)),
                    xycoords='data', rotation=90,
                    verticalalignment='bottom',
                    ha=annot_ha,
                    fontsize=10)

    return fig


plt.close('all')
fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=True)
axes = axes.ravel()
for ii, (case, key, title) in enumerate(cases):
    ax = axes[ii]
    unbiased_acc_list, lr_pvalues, coef_list, feat_names = results[ii]
    acc_offset = 0.1
    if case == 'case4':
        acc_offset = -0.35

    _plot_infpred(unbiased_acc_list, lr_pvalues, coef_list, feat_names,
                  acc_offset, ax=ax)

    ax.set_title(title,
                 fontsize=12, fontweight=150)
    ax.axvline(
        -np.log10(0.05), color='red', linestyle='--', linewidth=3)
    if case == 'case2':
        ax.annotate('p < 0.05', xy=(0.75, 0.03), color='red',
                    fontsize=14,)
    elif case == 'case4':
        ax.annotate('p < 0.05', xy=(5, 0.03), color='red',
                    fontsize=14,)
    else:
        ax.annotate('p < 0.05', xy=(-np.log10(0.045), 0.03), color='red',
                    fontsize=14,)
    ax.set_xlabel(r'significance [$-log_{10}(p)$]', fontsize=14,
                  fontweight=150)
    if ii == 0:
        ax.set_ylabel(r'prediction [$R^2$]', fontsize=14, fontweight=150)
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.set_yticks([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    ax.set_yticks(np.arange(0.01, 1, 0.01), minor=True)
    ax.annotate(
        'ABCD'[ii], xy=(-0.15, 1.05), fontweight=200, fontsize=30,
        xycoords='axes fraction')

plt.subplots_adjust(top=.83, bottom=0.15)
fig.savefig('./figures/reg-cases.pdf', bbox_inches='tight')
fig.savefig('./figures/reg-cases.png', bbox_inches='tight', dpi=300)
