# Authors: Denis Engemann <denis.engemann@gmail.com>
#          Danilo Bzdok <danilobzdok@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import pandas as pd
from sklearn import metrics

df = pd.read_pickle('./simulations_may.gzip')

# XXX Brute force!
# We assemble a matrix of true coefficients for each simulation
true_coef = np.ones((df.shape[0], 40)).astype(bool)
for n_rel, inds in df.groupby('n_feat_relevant').groups.items():
    true_coef[inds, n_rel:] = False


def get_confusion(truth, guess, prefix):
    """Convenience function."""
    cmat = metrics.confusion_matrix(truth, guess)
    # XXX potential sklearn bug: if all correct,
    # one single value is returned with matrix of shape(1,1)
    if len(np.unique(cmat)) == 1 and np.unique(cmat)[0] == 40:
        cmat = np.array((0, 0, 0, 40))
    # hand-checked.
    cmat = np.nan_to_num(cmat / cmat.sum())
    tn, fp, fn, tp = cmat.ravel()
    out = {f"tn_{prefix}": tn,
           f"fp_{prefix}": fp,
           f"fn_{prefix}": fn,
           f"tp_{prefix}": tp}
    return out


recon = list()
# XXX More brute force!
# Now we assess recovery for inference vs prediction.
for ii, row in df.iterrows():
    this_pvals_bonf = row['lr_pvalues'] * 40 < 0.05
    this_pvals_uncor = row['lr_pvalues'] < 0.05
    idx = row['scores_debiased'].argmax()
    this_lasso = row['coefs'][idx] != 0
    assert np.array_equal(this_lasso, np.abs(row['coefs'][idx]) > 0e-8)
    this_truth = true_coef[ii]
    rec = {
        'lasso_true': np.sum(this_truth == this_lasso),
        'ols_true_uncorr': np.sum(this_truth == this_pvals_uncor),
        'ols_true_bonf': np.sum(this_truth == this_pvals_bonf),
        'sim_id': row['sim_id']
    }
    rec.update(get_confusion(this_truth, this_pvals_uncor, "inf_uncor"))
    rec.update(get_confusion(this_truth, this_pvals_bonf, "inf_bonf"))
    rec.update(get_confusion(this_truth, this_lasso, "pred"))
    recon.append(rec)
recon = pd.DataFrame(recon)

recon.to_csv('./simultations_reconstruction_normalized.csv')

print((recon[['tn_pred', 'tp_pred', 'fn_pred', 'fp_pred']] == 0).sum())
print((recon[['tn_pred', 'tp_pred', 'fn_pred', 'fp_pred']] != 0).sum())
print((recon[['tn_inf_bonf', 'tp_inf_bonf', 'fn_inf_bonf', 'fp_inf_bonf']] == 0).sum())
print((recon[['tn_inf_bonf', 'tp_inf_bonf', 'fn_inf_bonf', 'fp_inf_bonf']] != 0).sum())
