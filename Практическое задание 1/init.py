from metrics import AUC, NUM, ASY1, ASY2
from utils import Dataset, make_histplot, LogisticRegressionValidate, CatBoostClassifierValidate, general_summary, opt_thresholds

#------------------------------

import matplotlib.pyplot as plt

myparams = {
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath}',
    'font.size': 14,
    #'axes.grid': True,
    #'grid.alpha': 0.2,
    'lines.linewidth': 2
}

plt.rcParams.update(myparams)

#------------------------------

import numpy as np
import pandas as pd
import seaborn as sns

#------------------------------

from catboost import CatBoostClassifier
from sklearn.metrics import RocCurveDisplay, make_scorer
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from scipy.stats import shapiro, norm, laplace
from statsmodels.api import qqplot
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


