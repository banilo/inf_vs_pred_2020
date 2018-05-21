# Authors: Denis Engemann <denis.engemann@gmail.com>
#          Danilo Bzdok <danilobzdok@gmail.com>
#
# License: BSD (3-clause)

import itertools

from joblib import Parallel, delayed

import numpy as np
import pandas as pd

from statsmodels.regression.linear_model import OLS


def _clip_pvals(pvals):
    # pvals[pvals == 0] = np.finfo(pvals.dtype).eps
    pvals[pvals == 0] = 2.2250738585072014e-308
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


attr_list = [
    'aic',
    'bic',
    'bse',
    'df_model',
    'df_resid',
    'ess',
    'llf',
    'mse_model',
    'mse_resid',
    'mse_total',
    'resid',
    'rsquared',
    'rsquared_adj',
    'ssr',
]


def run_simulation(sim_id, n_samples, n_feat, n_feat_relevant, noise_level,
                   correlation, model_violation):
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
    # Compute Laso regularization paths.
    # Check if simulation is useful and zero coefs occur.

    # Bundle results and good bye.
    print('Done', sim_id, n_samples, n_feat, n_feat_relevant, noise_level,
          correlation, model_violation)
    out = dict(
        n_samples=n_samples, n_feat=n_feat, n_feat_relevant=n_feat_relevant,
        lr_coefs=lr_coefs, lr_pvalues=lr_pvalues,
        model_violation=model_violation, sim_id=sim_id, noise=noise_level)
    out.update(correlation)
    for key in attr_list:
        out['res_' + key] = getattr(res, key)
    return out


iter_sim = itertools.product(
    sample_range, n_feat_range, n_feat_relevant_range,
    epsilon_range, correlation, model_violation)

out = Parallel(n_jobs=48)(
    delayed(run_simulation)(sim_id, *params)
    for sim_id, params in enumerate(iter_sim))

df = pd.DataFrame([oo for oo in out if oo is not None])
repr(df.loc[:10]['lr_pvalues'])

df.to_pickle('./simulations_ols.gzip')
