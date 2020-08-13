# Authors: Denis Engemann <denis.engemann@gmail.com>
#          Danilo Bzdok <danilobzdok@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from sklearn.model_selection import ShuffleSplit

rng = np.random.RandomState(35)

X = rng.normal(0, 1, size=(10, 4))
y = rng.lognormal(1, 1, size=(10, 1))

mpl.rc("font", **{"sans-serif": "Arial", "family": "sans-serif"})

plt.close("all")

blue = sns.light_palette("navy")
red = sns.light_palette("red")

fig1, ax = plt.subplots(1, 1, figsize=(3, 5))
sns.heatmap(X, cbar=False, ax=ax, annot=True, cmap=blue)
divider = make_axes_locatable(ax)
ax2 = divider.append_axes('right', size='25%', pad=0.5)
sns.heatmap(y, cbar=False, ax=ax2, annot=True, cmap=red)

ax.set_xticks([])
ax.set_yticks([])
ax2.set_xticks([])
ax2.set_yticks([])

cv = ShuffleSplit(random_state=3, test_size=.1)
train, test = next(cv.split(X, y))

fig2, ax = plt.subplots(1, 1, figsize=(3, 4.5))
sns.heatmap(X[train], cbar=False, ax=ax, annot=True, cmap=blue)
divider = make_axes_locatable(ax)
ax2 = divider.append_axes('right', size='25%', pad=0.5)
sns.heatmap(y[train], cbar=False, ax=ax2, annot=True, cmap=red)

ax.set_xticks([])
ax.set_yticks([])
ax2.set_xticks([])
ax2.set_yticks([])

fig3, ax = plt.subplots(1, 1, figsize=(3, 0.5))
sns.heatmap(X[test], cbar=False, ax=ax, annot=True, cmap=blue)
divider = make_axes_locatable(ax)
ax2 = divider.append_axes('right', size='25%', pad=0.5)
sns.heatmap(y[test], cbar=False, ax=ax2, annot=True, cmap=red)

ax.set_xticks([])
ax.set_yticks([])
ax2.set_xticks([])
ax2.set_yticks([])

fig1.savefig("figures/concept_element1.pdf", bbox_inches='tight')
fig2.savefig("figures/concept_element2.pdf", bbox_inches='tight')
fig3.savefig("figures/concept_element3.pdf", bbox_inches='tight')
