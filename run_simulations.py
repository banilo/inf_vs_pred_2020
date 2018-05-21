# Authors: Denis Engemann <denis.engemann@gmail.com>
#          Danilo Bzdok <danilobzdok@gmail.com>
#
# License: BSD (3-clause)

import itertools

from joblib import Parallel, delayed

import numpy as np
import pandas as pd

from sklearn.cross_validation import ShuffleSplit
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import r2_score
from statsmodels.regression.linear_model import OLS


def compute_lasso_regpath(X, y, C_grid, metric=None, verbose=True):
    """Run lass path and compute biased + debiased accuracy."""
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

            # get out-of-sample accuracy from unbiased linear model
            # with selected inputs
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
            print("alpha: %.4f R2: %.2f / %.2f (unbiased) "
                  "active_coefs: %i" % (
                      my_C, acc_for_C, acc_for_C_unbaised, notzero))
    out = (np.array(coef_list2),
           np.array(acc_list2),
           np.array(acc_unbiased_list2),
           np.array(nonzero_list2))
    return out


def _clip_pvals(pvals):
    pvals[pvals == 0] = np.finfo(pvals.dtype).eps
    return pvals


sample_range = list(np.arange(50, 100, 10))  # low n regime
sample_range += list(np.arange(100, 2001, 100))  # mid range
sample_range += [10000, 100000]  # biggish data

n_feat_range = (40,)
n_feat_relevant_range = list(np.arange(1, 41, 3))
epsilon_range = (0, 0.5, 1, 2, 5, 10)

correlation = tuple(
    [dict(n_corr_feat=x, corr_strength=y)
     for x in (0.5, 1.0)
     for y in (0.5, 0.9)])

correlation += (None,)  # add one case where we have no correlations

model_violation = (None, 'abs', 'log', 'exp', 'sqrt', '1/x',
                   'x^2', 'x^3', 'x^4', 'x^5')

C_grid = np.logspace(-3, 2, 50)

iter_sim = itertools.product(
    sample_range, n_feat_range, n_feat_relevant_range,
    epsilon_range, correlation, model_violation)


def run_simulation(sim_id, n_samples, n_feat, n_feat_relevant, noise_level,
                   correlation, model_violation,
                   C_grid=C_grid):
    """Run Inference-Prediction simulation."""
    # Set up ground truth model.
    print(sim_id)
    seed = sim_id
    # different random initialization for each sim.
    rs = np.random.RandomState(seed)
    # if noise_level==0 then no NOISE
    epsilon = rs.randn(n_samples) * noise_level
    true_coefs = rs.randn(n_feat)
    true_coefs[n_feat_relevant:] = 0
    X = rs.randn(n_samples, n_feat)

    # Introduce correlation:
    if correlation is not None and n_feat_relevant > 1:
        n_corr_feat = int(round(n_feat_relevant * correlation['n_corr_feat']))
        corr_strength = correlation['corr_strength']
        cov = np.ones((n_corr_feat, n_corr_feat)) * corr_strength
        cov.flat[::n_corr_feat + 1] = 1

        X_corr = rs.multivariate_normal(
            mean=np.zeros((n_corr_feat)), cov=cov, size=n_samples)
        X[:, 0:n_corr_feat] = X_corr
    else:
        correlation = dict(n_corr_feat=None, corr_strength=None)

    # Introduce transforms that are not captured by the model.
    if model_violation is None:
        X_viol = X.copy()
    elif model_violation is not None:
        if n_feat_relevant > 0:
            n_viol = max(1, int(round(n_feat_relevant / 2)))
        else:
            print('irrelevant scenario')
            return None  # nothing to do ... case already covered.

        X_viol = X.copy()
        signs = np.sign(X)
        if model_violation == 'abs':
            X_viol[:, :n_viol] = np.abs(X_viol[:, :n_viol])
        elif model_violation == 'log':
            X_viol[:, :n_viol] = signs[:, :n_viol] * np.log(
                np.abs(X_viol[:, :n_viol]))
        elif model_violation == 'sqrt':
            X_viol[:, :n_viol] = signs[:, :n_viol] * np.sqrt(
                np.abs(X_viol[:, :n_viol]))
        elif model_violation == 'exp':
            X_viol[:, :n_viol] = signs[:, :n_viol] * np.exp(X_viol[:, :n_viol])
        elif model_violation == '1/x':
            X_viol[:, :n_viol] = 1. / X_viol[:, :n_viol]
        elif model_violation == 'x^2':
            X_viol[:, :n_viol] **= 2
        elif model_violation == 'x^3':
            X_viol[:, :n_viol] **= 3
        elif model_violation == 'x^4':
            X_viol[:, :n_viol] **= 4
        elif model_violation == 'x^5':
            X_viol[:, :n_viol] **= 5

    # Generate target.
    y = (true_coefs * X_viol).sum(axis=1) + epsilon

    # Compute ordinary least squares.
    model = OLS(y, X)
    res = model.fit()
    lr_coefs = res.params
    lr_pvalues = _clip_pvals(res.pvalues)
    # Compute Lasso regularization paths.
    coefs, scores, scores_debiased, nonzero = compute_lasso_regpath(
        X, y, C_grid)

    # Check if simulation is useful and zero coefs occur.
    C_grid_is_success = True
    if np.min(nonzero) > 0:
        C_grid_is_success = False
        print('Bad')

    # Bundle results and good bye.
    print('Done')
    out = dict(
        n_samples=n_samples, n_feat=n_feat, n_feat_relevant=n_feat_relevant,
        lr_coefs=lr_coefs, lr_pvalues=lr_pvalues,
        coefs=coefs, nonzero=nonzero,
        scores=scores, scores_debiased=scores_debiased,
        model_violation=model_violation, sim_id=sim_id, noise=noise_level,
        C_grid_is_success=C_grid_is_success)
    out.update(correlation)
    return out

out = Parallel(n_jobs=1)(
    delayed(run_simulation)(sim_id, *params)
    for sim_id, params in enumerate(iter_sim) if sim_id == 0)

df = pd.DataFrame([oo for oo in out if oo is not None])

df.to_pickle('./simulations.gzip')
