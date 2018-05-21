[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/banilo/inf_vs_pred_2018/master)

# Inference vs prediction

Code, interactive visualization and data for the paper:

Prediction and inference diverge in biomedicine:
Simulations and real-world data

Danilo Bzdok, Denis A. Engemann, Olivier Grisel, Gaël Varoquaux, Bertrand Thirion


## Interactive exploration of analysis and supplementary materials

For interactive exploration launch a binder session by clicking on the badge above.

- ```infpred_interactive_simu_explorer.ipynb``` provides an interactive visualization tool for exploring the simulation results reported in the paper (Figures 1-3).

- ```infpred_appl_classif.ipynb``` provides a detailed tutorial on exploring the medical data analyzed in the paper (Figure 4).

- ```infpred_appl_regr.ipynb``` extends the analysis of medical data to classification problems.

- ```infpred_simu.ipynb``` presents in greater detail exploration of selected simulation scenarios.

Note. As computational resources are limited in Binder certain cells of the notebook may take longer to run, especially for visualization using Seaborn. You can skip those cells or have to be patient.

Depending on your internet connection and current traffic, Binder may take more or less time to load the interactive session.

For questions about Binder consider the webpage of the project (https://mybinder.org).

## Local exploration of analysis and supplementary materials

Alternatively, the analyses can also be explored locally by cloning the git repository. To be able to run the code make sure to have a Python installation satisfying the dependencies declared in ```environment.yaml```: numpy, scipy, pandas, statsmodels, scikit-Learn, matplotlib and seaborn.
