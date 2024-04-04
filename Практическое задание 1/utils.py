import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate, cross_val_predict
from prettytable import PrettyTable
from sklearn.metrics import make_scorer
from scipy.stats import norm
from metrics import AUC, Accuracy, NUM, ASY1, RASY1, ASY2, RASY2
import os
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
import tqdm


def D_KL(m_0, std_0, m_1, std_1):
    """
    Calculate the Kullback-Leibler divergence between two normal univariate distributions,
    i.e. D_KL( N(m_0, std_0) || N(m_1, std_1) )

    Parameters
    ----------
    m_0: float
        Mathematical expectation of N_0.

    sdt_0: float
        Standard deviation of N_0.

    m_1: float
        Mathematical expectation of N_1.

    sdt_1: float
        Standard deviation of N_1.

    Returns
    -------
    D_KL( N(m_0, std_0) || N(m_1, std_1) ): float
        Kullback-Leibler divergence between two normal univariate distributions.
    """

    return np.log(std_1/std_0) + (std_0**2 + (m_0 - m_1)**2) / (2 * std_1**2) - 1/2


def make_histplot(df, figsize=(19, 19)):
    m_1 = None
    num_features = int(df.shape[1]) - 1
    for m in range(1, num_features+1):
        if num_features % m == 0:
            if (num_features // m <= m):
                break
            else:
                m_1 = m
    fig, ax = plt.subplots(num_features // m_1, m_1, figsize=(19, 19))

    for i in range(num_features // m_1):
        for j in range(m_1):
            sns.histplot(data=df, x=df.columns[m_1 * i + j],
                         stat='density', hue='Class', kde=True, edgecolor=None, ax=ax[i, j])

            ax[i, j].grid(alpha=0.2)

    plt.tight_layout()
    plt.show()


def general_summary():

    x = PrettyTable()
    x.field_names = ['Выборка', 'Обучающая, # объектов',
                     'Тестовая, # объектов', '# признаков', 'Доля класса 1, %']

    for dataset_id in range(1, 15):

        dataset = Dataset(dataset_id)
        x.add_row([dataset.dataset_id, dataset.n_samples, dataset.X2.shape[0], dataset.n_features,
                   np.round(dataset.y1.sum() / len(dataset.y1) * 100, 1)])

    print(x)


class Dataset:

    def __init__(self, dataset_id):

        self.dataset_id = dataset_id

        self.X1_path = f'./data/task1_{dataset_id}_learn_X.csv'
        self.y1_path = f'./data/task1_{dataset_id}_learn_y.csv'
        self.X2_path = f'./data/task1_{dataset_id}_test_X.csv'

        self.get_data()
        self.get_scaled_data()
        self.get_divergences(verbose=False)

    def summary(self):

        x = PrettyTable()
        x.field_names = ['Выборка', 'Обучающая, # объектов',
                         'Тестовая, # объектов', '# признаков', 'Доля класса 1, %']
        x.add_row([self.dataset_id, self.n_samples, self.X2.shape[0],
                  self.n_features, np.round(self.y1.sum() / len(self.y1) * 100, 1)])
        print(x)

    def get_data(self):

        self.X1 = pd.read_csv(self.X1_path, sep=' ', header=None)
        self.y1 = pd.read_csv(self.y1_path, header=None, names=[
                              'Class']).Class.astype(int)
        self.X2 = pd.read_csv(self.X2_path, sep=' ', header=None)

        self.n_samples = self.X1.shape[0]
        self.n_features = self.X1.shape[1]

        self.df = pd.concat([self.X1, self.y1], axis=1)

    def save_ans(self, models, features_list, cols_list, thresholds_list):

        if len(models) != 4:  # one model for each metric is needed
            raise ValueError('We need 4 models to save answers')

        for model, features in zip(models, features_list):
            try:
                model.fit(features, self.y1, verbose=False)
            except TypeError:
                model.fit(features, self.y1)

        y_score_auc = models[0].predict_proba(
            self.X2_scaled.iloc[:, cols_list[0]])[:, 1]
        y_score_num = models[1].predict_proba(
            self.X2_scaled.iloc[:, cols_list[1]])[:, 1]
        y_score_asy1 = models[2].predict_proba(
            self.X2_scaled.iloc[:, cols_list[2]])[:, 1]
        y_score_asy2 = models[3].predict_proba(
            self.X2_scaled.iloc[:, cols_list[3]])[:, 1]

        y_pred_num = np.vectorize(lambda p: 1 if p > thresholds_list[0] else 0)(y_score_num)
        y_pred_asy1 = np.vectorize(lambda p: 1 if p > thresholds_list[1] else 0)(y_score_asy1)
        y_pred_asy2 = np.vectorize(lambda p: 1 if p > thresholds_list[2] else 0)(y_score_asy2)

        predictions = np.array([y_score_auc, y_pred_num, y_pred_asy1, y_pred_asy2], dtype=object)
        out = pd.DataFrame(data=predictions.T, columns=['AUC', 'NUM', 'ASY1', 'ASY2'])

        if not os.path.exists('ans'):
            os.makedirs('ans')

        out.to_csv(f'./ans/task1_{self.dataset_id}_ans.csv', index=False)

    def get_scaled_data(self):
        scaler = StandardScaler()
        scaler.fit(self.X1)
        self.X1_scaled = pd.DataFrame(scaler.transform(self.X1))
        self.X2_scaled = pd.DataFrame(scaler.transform(self.X2))
        self.df_scaled = pd.concat([self.X1_scaled, self.y1], axis=1)
        self.X_train_scaled = self.X1_scaled.values
        self.y_train = self.y1.values.flatten()
        self.X_test_scaled = self.X2_scaled.values

    def get_scoring(self, t_Accuracy, t_RASY1, t_RASY2):

        self.scoring = {
            'AUC': AUC_scorer,
            'Accuracy': Accuracy_scorer_threshold(t_Accuracy),
            'RASY1': RASY1_scorer_threshold(t_RASY1),
            'RASY2': RASY2_scorer_threshold(t_RASY2)
        }

        self.scoring_final = {
            'AUC': AUC_scorer,
            'NUM': NUM_scorer_threshold(t_Accuracy),
            'ASY1': ASY1_scorer_threshold(t_RASY1),
            'ASY2': ASY2_scorer_threshold(t_RASY2)
        }

    def heatmap(self, label='train', save=False):
        #sns.set(rc={"figure.figsize":(12, 10)})
        #sns.set(font_scale= 1.2)
        if label == 'train':
            sns.heatmap(self.X1_scaled.corr(), vmin=-1, vmax=1,
                        center=0, cmap='vlag', square=True)
        elif label == 'test':
            sns.heatmap(self.X2_scaled.corr(), vmin=-1, vmax=1,
                        center=0, cmap='vlag', square=True)
        plt.tight_layout()
        if save == True:
            plt.savefig(
                f'./fig/dataset_{self.dataset_id}/heatmap.pdf', bbox_inches='tight')
        plt.show()

    def target_corr(self, annot=True, save=False):
        sns.heatmap(self.df_scaled.corr().Class.values.reshape(-1, 1),
                    annot=annot, fmt='.2f', vmin=-1, vmax=1, center=0, cmap='vlag')
        plt.tight_layout()
        if save == True:
            plt.savefig(
                f'./fig/dataset_{self.dataset_id}/target_corr.pdf', bbox_inches='tight')
        plt.show()

    def approx(self, number_of_feature, save=False):

        mu_1, std_1 = norm.fit(self.X1_scaled[number_of_feature][self.y1 == 1])
        print(f'Признак номер {number_of_feature}')
        print('\nКласс 1:')
        print("Среднее значение (mu):", mu_1)
        print("Стандартное отклонение (std):", std_1)
        mu_0, std_0 = norm.fit(self.X1_scaled[number_of_feature][self.y1 == 0])
        print('\nКласс 0:')
        print("Среднее значение (mu):", mu_0)
        print("Стандартное отклонение (std):", std_0)

        sns.histplot(data=self.X1_scaled[number_of_feature][self.y1 == 0],
                     color='blue', stat='density', edgecolor=None, alpha=0.5)
        sns.histplot(data=self.X1_scaled[number_of_feature][self.y1 == 1],
                     color='coral', stat='density', edgecolor=None, alpha=0.5)

        grid = np.linspace(min(self.X1_scaled[number_of_feature]), max(
            self.X1_scaled[number_of_feature]), 10000)
        plt.plot(grid, norm.pdf(grid, loc=mu_0, scale=std_0),
                 linewidth=3, label='Class 0')
        plt.plot(grid, norm.pdf(grid, loc=mu_1, scale=std_1),
                 linewidth=3, label='Class 1')

        plt.legend()

        plt.grid(alpha=0.2)
        if save == True:
            plt.savefig(
                f'./fig/dataset_{self.dataset_id}/approx_{number_of_feature}.pdf', bbox_inches='tight')
        plt.tight_layout()
        plt.show()

    def feature_selection(self, n_cols=None):

        cols = np.argsort(self.divergences)[::-1][:n_cols]
        X = self.X1_scaled.iloc[:, cols]

        return X, cols

    def evaluate(self, model, X=None, verbose=True, relative=True, fit_params=None):

        if X is None:
            X = self.X1_scaled

        # каждое y_j получается, когда оно содержится в тестовой выборке на кросс-валидации
        y_score = cross_val_predict(model, X, self.y1, method='predict_proba')[:, 1]

        output = opt_thresholds(self.y1, y_score, visualize=False)

        t_Accuracy = output['Accuracy']['t']
        t_RASY1 = output['RASY1']['t']
        t_RASY2 = output['RASY2']['t']

        self.get_scoring(t_Accuracy, t_RASY1, t_RASY2)

        if relative == True:
            scoring = self.scoring
            sign = 1
        else:
            scoring = self.scoring_final
            sign = -1

        scores = cross_validate(model, X, self.y1, scoring=scoring, fit_params=fit_params)
        metrics = list(scoring.keys())
        output = {
            metrics[0]: {'score': round(scores['test_' + metrics[0]].mean(), 4)},
            metrics[1]: {'t': t_Accuracy, 'score': round(sign * scores['test_' + metrics[1]].mean(), 4)},
            metrics[2]: {'t': t_RASY1, 'score': round(-scores['test_' + metrics[2]].mean(), 4)},
            metrics[3]: {'t': t_RASY2, 'score': round(-scores['test_' + metrics[3]].mean(), 4)}
        }

        if verbose == True:

            x = PrettyTable()

            x.field_names = ["Metric", "Cross-Validation"]

            for i in range(4):
                x.add_row([metrics[i], output[metrics[i]]['score']])

            print(x)

        return output

    def get_divergences(self, verbose=False):

        divergences = []

        for j in range(self.n_features):

            x0 = self.X1_scaled[self.y1 == 0][j]
            x1 = self.X1_scaled[self.y1 == 1][j]
            m_0, std_0 = norm.fit(x0)
            m_1, std_1 = norm.fit(x1)
            divergence = D_KL(m_0, std_0, m_1, std_1)
            if verbose == True:
                print(
                    f'KL-divergence for feature #{j}: {divergence:.3f}, Class 1: N({m_1:.3f}, {std_1:.3f}), Class 0: N({m_0:.3f}, {std_0:.3f})')
            divergences.append(divergence)

        divergences = np.array(divergences)
        self.divergences = divergences

    def metrics_n_features_dependence(self, model, fit_params=None, ks=None, save=False):

        if ks is None:
            ks = np.arange(1, self.n_features+1)

        thresholds_matrix = []
        metrics_matrix = []
        labels = ['AUC', 'Accuracy', 'RASY1', 'RASY2']

        for k in tqdm.tqdm(ks):
            X, _ = self.feature_selection(n_cols=k)
            output = self.evaluate(model, X=X, verbose=False, relative=True, fit_params=None)
            thresholds = np.array([output[metric]['t'] for metric in labels[1:]])
            thresholds_matrix.append(thresholds)
            metrics = np.array([output[metric]['score'] for metric in labels])
            metrics_matrix.append(metrics)

        thresholds_matrix = np.array(thresholds_matrix)
        metrics_matrix = np.array(metrics_matrix)

        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        colors = ['royalblue', 'firebrick', 'forestgreen', 'darkorange']

        k_AUC = np.argmax(metrics_matrix[:, 0]) + 1
        k_Accuracy = np.argmax(metrics_matrix[:, 1]) + 1
        k_RASY1 = np.argmin(metrics_matrix[:, 2]) + 1
        k_RASY2 = np.argmin(metrics_matrix[:, 3]) + 1

        features_AUC, cols_AUC = self.feature_selection(n_cols=k_AUC)
        features_Accuracy, cols_Accuracy = self.feature_selection(n_cols=k_Accuracy)
        features_RASY1, cols_RASY1 = self.feature_selection(n_cols=k_RASY1)
        features_RASY2, cols_RASY2 = self.feature_selection(n_cols=k_RASY2)

        t_Accuracy = thresholds_matrix[:, 0][k_Accuracy-1]
        t_RASY1 = thresholds_matrix[:, 1][k_RASY1-1]
        t_RASY2 = thresholds_matrix[:, 2][k_RASY2-1]

        for i in range(2):
            for j in range(2):
                ax[i, j].plot(ks, metrics_matrix[:, 2*i+j],
                              color=colors[2*i+j], label=labels[2*i+j])
                ax[i, j].legend()
                ax[i, j].set_xlabel('Number of features')
                ax[i, j].grid(alpha=0.2)

        plt.tight_layout()
        if save == True:
            plt.savefig(
                f'./fig/dataset_{self.dataset_id}/metrics_n_features_dependence.pdf', bbox_inches='tight')
        plt.show()

        output = {'AUC': {'features': features_AUC, 'cols': cols_AUC},
                  'Accuracy': {'features': features_Accuracy, 'cols': cols_Accuracy, 't': t_Accuracy},
                  'RASY1': {'features': features_RASY1, 'cols': cols_RASY1, 't': t_RASY1},
                  'RASY2': {'features': features_RASY2, 'cols': cols_RASY2, 't': t_RASY2}}

        return output

def LogisticRegressionValidate(dataset, features_cols_lr, Cs, save=False):

    features = {
        'AUC': features_cols_lr['AUC']['features'],
        'Accuracy': features_cols_lr['Accuracy']['features'],
        'RASY1': features_cols_lr['RASY1']['features'],
        'RASY2': features_cols_lr['RASY2']['features']
    }

    labels = ['AUC', 'Accuracy', 'RASY1', 'RASY2']
    thresholds_matrix = []
    metrics_matrix = []

    for C in tqdm.tqdm(Cs):

        thresholds = []
        metrics = []

        for i, metric in enumerate(labels):

            model = LogisticRegression(C=C)
            output = dataset.evaluate(model, X=features[metric], verbose=False, relative=True, fit_params=None)[metric]
            if i > 0:
                thresholds.append(output['t'])
            metrics.append(output['score'])
            
        thresholds = np.array(thresholds)
        metrics = np.array(metrics)

        thresholds_matrix.append(thresholds)
        metrics_matrix.append(metrics)

    thresholds_matrix = np.array(thresholds_matrix)
    metrics_matrix = np.array(metrics_matrix)
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    colors = ['royalblue', 'firebrick', 'forestgreen', 'darkorange']

    C_idx_AUC = np.argmax(metrics_matrix[:, 0])
    C_idx_Accuracy = np.argmax(metrics_matrix[:, 1])
    C_idx_RASY1 = np.argmin(metrics_matrix[:, 2])
    C_idx_RASY2 = np.argmin(metrics_matrix[:, 3])

    C_AUC = Cs[C_idx_AUC]
    C_Accuracy = Cs[C_idx_Accuracy]
    C_RASY1 = Cs[C_idx_RASY1]
    C_RASY2 = Cs[C_idx_RASY2]

    t_Accuracy = thresholds_matrix[:, 0][C_idx_Accuracy]
    t_RASY1 = thresholds_matrix[:, 1][C_idx_RASY1]
    t_RASY2 = thresholds_matrix[:, 2][C_idx_RASY2]

    for i in range(2):
        for j in range(2):
            ax[i, j].plot(Cs, metrics_matrix[:, 2*i+j],
                          color=colors[2*i+j], label=labels[2*i+j])
            ax[i, j].legend()
            ax[i, j].set_xscale('log')
            ax[i, j].grid(alpha=0.2)

    plt.tight_layout()
    if save == True:
            plt.savefig(
                f'./fig/dataset_{dataset.dataset_id}/LogisticRegressionValidate.pdf', bbox_inches='tight')
    plt.show()

    output = {
        'AUC': {'C': C_AUC},
        'Accuracy': {'t': t_Accuracy, 'C': C_Accuracy},
        'RASY1': {'t': t_RASY1, 'C': C_RASY1},
        'RASY2': {'t': t_RASY2, 'C': C_RASY2}
    }

    return output


def CatBoostClassifierValidate(dataset, features, depths, fit_params={"verbose": False, "plot": False}):

    metrics_matrix = []

    for depth in depths:
        model = CatBoostClassifier(depth=depth)
        results = dataset.evaluate(model, verbose=False, X=features,
                                   fit_params=fit_params)
        metrics_matrix.append(results)

    metrics_matrix = np.array(metrics_matrix)
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    labels = ['AUC', 'Accuracy', 'RASY1', 'RASY2']
    colors = ['royalblue', 'firebrick', 'forestgreen', 'darkorange']

    for i in range(2):
        for j in range(2):
            ax[i, j].plot(depths, metrics_matrix[:, 2*i+j],
                          color=colors[2*i+j], label=labels[2*i+j])
            ax[i, j].legend()
            ax[i, j].set_xscale('log')
            ax[i, j].grid(alpha=0.2)

    plt.tight_layout()
    plt.show()

def opt_threshold_metric(metric, y_true, y_score, greater_is_better=True, visualize=True):
    """
    Optimize treshold for binary classification for given metric.

    Parameters
    ----------
    metric:
        Metric to be optimized.

    y_true: array-like of shape (n_samples,)
        True labels.

    y_score: array-like of shape (n_samples,)
        Target scores.

    visualize: bool
        Whether to plot results or not.

    Returns
    -------
    y_pred:
        Optimum prediction.

    """

    if greater_is_better == True:
        sign = 1
    else:
        sign = -1

    metric_list = []
    ts = np.linspace(0, 1, 100)

    for t in ts:

        probability = np.vectorize(lambda p: 1 if p > t else 0)
        y_pred = probability(y_score)
        metric_list.append(metric(y_true, y_pred))

    metric_list = np.array(metric_list)

    t_idx = np.argmax(sign * metric_list)
    t = ts[t_idx]

    probability = np.vectorize(lambda p: 1 if p > t else 0)
    y_pred = probability(y_score)

    if visualize == True:

        plt.plot(ts, metric_list, color='royalblue')
        plt.scatter(t, metric[t_idx], marker='x', s=50, color='black', label=f't = {t:0.2f}')
        plt.plot(ts[:t_idx], metric[t_idx] * np.ones(t_idx), linestyle='dashed', linewidth=1, color='black')
        plt.legend()
        plt.set_xlabel('Threshold')
        plt.grid(alpha=0.2)

        plt.tight_layout()
        plt.show()

    return y_pred


def opt_thresholds(y_true, y_score, visualize=True, save=False):

    Accuracy_list = [] 
    RASY1_list = []
    RASY2_list = []

    ts = np.linspace(0, 1, 100)

    for t in ts:

        probability = np.vectorize(lambda p: 1 if p > t else 0)
        y_pred = probability(y_score)
        Accuracy_list.append(Accuracy(y_true, y_pred))
        RASY1_list.append(RASY1(y_true, y_pred))
        RASY2_list.append(RASY2(y_true, y_pred))

    t_idx_Accuracy = np.argmax(Accuracy_list)
    t_idx_RASY1 = np.argmin(RASY1_list)
    t_idx_RASY2 = np.argmin(RASY2_list)

    t_Accuracy = ts[t_idx_Accuracy]
    t_RASY1 = ts[t_idx_RASY1]
    t_RASY2 = ts[t_idx_RASY2]

    y_preds = []

    for t in [t_Accuracy, t_RASY1, t_RASY2]:

        probability = np.vectorize(lambda p: 1 if p > t else 0)
        y_preds.append(probability(y_score))

    if visualize == True:

        fig, ax = plt.subplots(1, 3, figsize=(14, 4))
        labels = ['Accuracy', 'RASY1', 'RASY2']
        colors = ['firebrick', 'forestgreen', 'darkorange']

        for i, metric, t_idx, label, color in zip(range(3), [Accuracy_list, RASY1_list, RASY2_list],
                                                [t_idx_Accuracy, t_idx_RASY1, t_idx_RASY2], labels, colors):
            ax[i].plot(ts, metric, color=color, label=label)
            ax[i].scatter(ts[t_idx], metric[t_idx], marker='x', s=50, color='black', label=f't = {ts[t_idx]:0.2f}')
            ax[i].plot(ts[:t_idx], metric[t_idx] * np.ones(t_idx), linestyle='dashed', linewidth=1, color='black')
            ax[i].legend()
            ax[i].set_xlabel('Threshold')
            ax[i].grid(alpha=0.2)

        plt.tight_layout()
        if save == True:
            plt.savefig(
                f'./fig/dataset_1/opt_thresholds.pdf', bbox_inches='tight')
        plt.show()

    output = {
        'Accuracy': {'t': t_Accuracy, 'y_pred': y_preds[0]},
        'RASY1': {'t': t_RASY1, 'y_pred': y_preds[1]},
        'RASY2': {'t': t_RASY2, 'y_pred': y_preds[2]},
    }

    return output

def AUC_scorer(clf, X, y):
    y_score = clf.predict_proba(X)[:, 1]
    return AUC(y, y_score)

def NUM_scorer_threshold(t):
    def NUM_scorer(clf, X, y):
        y_score = clf.predict_proba(X)[:, 1]
        y_pred = np.vectorize(lambda p: 1 if p > t else 0)(y_score)
        return -NUM(y, y_pred)
    return NUM_scorer

def Accuracy_scorer_threshold(t):
    def Accuracy_scorer(clf, X, y):
        y_score = clf.predict_proba(X)[:, 1]
        y_pred = np.vectorize(lambda p: 1 if p > t else 0)(y_score)
        return Accuracy(y, y_pred)
    return Accuracy_scorer

def ASY1_scorer_threshold(t):
    def ASY1_scorer(clf, X, y):
        y_score = clf.predict_proba(X)[:, 1]
        y_pred = np.vectorize(lambda p: 1 if p > t else 0)(y_score)
        return -ASY1(y, y_pred)
    return ASY1_scorer

def RASY1_scorer_threshold(t):
    def RASY1_scorer(clf, X, y):
        y_score = clf.predict_proba(X)[:, 1]
        y_pred = np.vectorize(lambda p: 1 if p > t else 0)(y_score)
        return -RASY1(y, y_pred)
    return RASY1_scorer

def ASY2_scorer_threshold(t):
    def ASY2_scorer(clf, X, y):
        y_score = clf.predict_proba(X)[:, 1]
        y_pred = np.vectorize(lambda p: 1 if p > t else 0)(y_score)
        return -ASY2(y, y_pred)
    return ASY2_scorer

def RASY2_scorer_threshold(t):
    def RASY2_scorer(clf, X, y):
        y_score = clf.predict_proba(X)[:, 1]
        y_pred = np.vectorize(lambda p: 1 if p > t else 0)(y_score)
        return -RASY2(y, y_pred)
    return RASY2_scorer
