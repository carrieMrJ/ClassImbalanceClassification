import smote_variants
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from imblearn.ensemble import RUSBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from costcla.models import CostSensitiveDecisionTreeClassifier
import evaluate_metrics
from models.CalibratedAdaMEC import CalibratedAdaMECClassifier
from models.MetaCost import MetaCost
from DDAE.ddae_main import MainDDAE
from iterative_metric.iml_model import IterativeMetricLearning
from models.self_paced_ensemble import SelfPacedEnsemble


def ddae(X_train, y_train, X_test, y_test, label, tup):
    y_train = y_train.map({tup[0]: '0', tup[1]: '1'})
    y_test = y_test.map({tup[0]: '0', tup[1]: '1'})

    model = MainDDAE(lb_column=label, maj_label='0', min_label='1',
                     n_neighbors=5, max_iter=50)

    model.fit(X_train, y_train)
    predict_ddae = model.predict(X_test)
    predict_proba = model.predict_proba(X_test)
    evaluate_metrics.ev_me(y_test, predict_ddae, ['0', '1'])
    evaluate_metrics.draw_apr_c(y_test, predict_proba, '1', 'DDAE')
    evaluate_metrics.draw_roc_c(y_test, predict_proba, '1', 'DDAE')


def rusboost(X_train, y_train, X_test, y_test):
    model = RUSBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=15,
                                                                     max_leaf_nodes=9,
                                                                     splitter='random'),
                               learning_rate=0.55, n_estimators=70, random_state=1234,
                               replacement=True)

    model.fit(X_train, y_train)

    predict_ada = model.predict(X_test)
    y_score_ada = model.predict_proba(X_test)

    evaluate_metrics.ev_me(y_test, predict_ada, [0, 1])

    evaluate_metrics.draw_apr_c(y_test, y_score_ada, 1, 'RUSBoost')
    evaluate_metrics.draw_roc_c(y_test, y_score_ada, 1, 'RUSBoost')

    return predict_ada


def mwmote(X_train, y_train, X_test, y_test):
    model = smote_variants.MWMOTE(random_state=1234)
    y_train = np.ravel(y_train)

    X_train, y_train = model.fit_resample(np.array(X_train), y_train)

    X_train = pd.DataFrame(X_train)
    y_train = pd.DataFrame(y_train, columns=['ClassValue'])

    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)

    knn_clf = KNeighborsClassifier(n_neighbors=2)

    knn_clf.fit(X_train, y_train)

    predict_mwmote = knn_clf.predict(X_test)
    y_score_mwmote = knn_clf.predict_proba(X_test)

    evaluate_metrics.ev_me(y_test, predict_mwmote, [0, 1])
    evaluate_metrics.draw_apr_c(y_test, y_score_mwmote, 1, 'MWMOTE')
    evaluate_metrics.draw_roc_c(y_test, y_score_mwmote, 1, 'MWMOTE')
    return predict_mwmote


def smote(X_train, y_train, X_test, y_test):
    model = SMOTE(random_state=1234)

    X_train, y_train = model.fit_resample(np.array(X_train), y_train)

    X_train = pd.DataFrame(X_train, columns=X.columns)

    y_test = np.ravel(y_test)
    y_train = np.ravel(y_train)

    knn_clf = KNeighborsClassifier(n_neighbors=2)

    knn_clf.fit(X_train, y_train)

    predict_mwmote = knn_clf.predict(X_test)
    y_score_mwmote = knn_clf.predict_proba(X_test)

    evaluate_metrics.ev_me(y_test, predict_mwmote, [0, 1])
    evaluate_metrics.draw_apr_c(y_test, y_score_mwmote, 1, 'SMOTE')
    evaluate_metrics.draw_roc_c(y_test, y_score_mwmote, 1, 'SMOTE')
    return predict_mwmote


def metacost(X_train, y_train, X_test, y_test, C_FN, label):
    y_train.columns = [label]
    y_test.columns = [label]

    C_FP = 1
    cost_matrix = np.array([[0, C_FN], [C_FP, 0]])

    model = MetaCost(pd.concat([X_train, y_train], axis=1), DecisionTreeClassifier(), cost_matrix).fit(label, 2)

    predict_meta = model.predict(X_test)
    y_score_meta = model.predict_proba(X_test)

    evaluate_metrics.ev_me(y_test, predict_meta, [0, 1])
    evaluate_metrics.draw_apr_c(y_test, y_score_meta, 1, 'MetaCost')
    evaluate_metrics.draw_roc_c(y_test, y_score_meta, 1, 'MetaCost')

    return predict_meta


def adaBoost(X_train, y_train, X_test, y_test):
    base_estimator = DecisionTreeClassifier(criterion='gini', max_depth=15, max_leaf_nodes=9, min_samples_split=2,
                                            splitter='random')

    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)

    model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=15,
                                                                     max_leaf_nodes=9,
                                                                     splitter='random'),
                               learning_rate=0.52, n_estimators=90, random_state=1234)

    model.fit(X_train, y_train)
    predict_ada = model.predict(X_test)
    y_score_ada = model.predict_proba(X_test)
    evaluate_metrics.ev_me(y_test, predict_ada, [0, 1])
    evaluate_metrics.draw_apr_c(y_test, y_score_ada, 1, 'AdaBoost')
    evaluate_metrics.draw_roc_c(y_test, y_score_ada, 1, 'AdaBoost')

    return predict_ada


def cadamec(X_train, y_train, X_test, y_test, C_FN, n_estimators):
    base_estimator = DecisionTreeClassifier(criterion='gini', max_depth=15, max_leaf_nodes=9,
                                            min_samples_split=2,
                                            splitter='random')

    C_FP = 1

    CaAdaMEC = CalibratedAdaMECClassifier(base_estimator, n_estimators, C_FP, C_FN)
    CaAdaMEC.fit(np.array(X_train), np.array(y_train))
    predict_ca = CaAdaMEC.predict(np.array(X_test))
    y_score_ca = CaAdaMEC.predict_proba((np.array(X_test)))

    evaluate_metrics.ev_me(y_test, predict_ca, [0, 1])
    evaluate_metrics.draw_apr_c(y_test, y_score_ca, 1, 'CAdaMEC')
    evaluate_metrics.draw_roc_c(y_test, y_score_ca, 1, 'CAdaMEC')

    return predict_ca


def costsensitivedct(X_train, y_train, X_test, y_test, C_FN):
    dct = CostSensitiveDecisionTreeClassifier()
    C_FP = 1

    cost_matrix = np.zeros((X_train.shape[0], 4))
    for k in range(X_train.shape[0]):
        cost_matrix[k, :] = np.array([C_FP, C_FN, 0, 0])

    dct.fit(np.array(X_train), y_train, cost_matrix)
    predict_csdct = dct.predict(np.array(X_test))
    y_score = dct.predict_proba(np.array(X_test))

    evaluate_metrics.ev_me(y_test, predict_csdct, [0, 1])
    evaluate_metrics.draw_apr_c(y_test, y_score, 1, 'csDCT')
    evaluate_metrics.draw_roc_c(y_test, y_score, 1, 'csDCT')


def self_paced(X_train, y_train, X_test, y_test):
    def absolute_error(y_true, y_pred):
        """Self-defined classification hardness function"""
        return np.absolute(y_true - y_pred)

    base_estimator = DecisionTreeClassifier(criterion='gini', max_depth=8, max_leaf_nodes=9,
                                            min_samples_split=2,
                                            splitter='random')

    model = SelfPacedEnsemble(base_estimator=base_estimator, hardness_func=absolute_error,
                              n_estimators=10).fit(np.array(X_train), np.array(y_train))
    predict_spe = model.predict(np.array(X_test))
    y_score = model.predict_proba(np.array(X_test))

    evaluate_metrics.ev_me(y_test, predict_spe, [0, 1])
    fig_pr = evaluate_metrics.draw_apr_c(y_test, y_score, 1, 'self-paced ensmeble')
    fig_roc = evaluate_metrics.draw_roc_c(y_test, y_score, 1, 'self-paced ensmeble')


def iml(X_train, y_train, X_test, y_test, lb, tup):
    y_train = y_train.map({tup[0]: '0', tup[1]: '1'})
    y_test = y_test.map({tup[0]: '0', tup[1]: '1'})

    model = IterativeMetricLearning(target_name=lb,
                                    n_neighbors=3,
                                    regularization=0.2,
                                    base_classifier=KNeighborsClassifier(n_neighbors=3),
                                    top_negative_number=5,
                                    top_positive_number=6,
                                    label_tupel=('0', '1'))
    model.fit(X_train, y_train)
    predict_iml = model.predict(X_test)

    predict_proba = model.predict_proba()

    evaluate_metrics.ev_me(y_test, predict_iml, ['0', '1'])
    evaluate_metrics.draw_apr_c(y_test, predict_proba, '1', 'IML')
    evaluate_metrics.draw_roc_c(y_test, predict_proba, '1', 'IML')


if __name__ == '__main__':

    data = pd.read_csv(r'datasets\mammography.csv')
    X = data.iloc[:, :-1]
    Y = data.iloc[:, -1]
    lb = data.columns[-1]
    X_tra, X_te, y_tra, y_te = train_test_split(X, Y, test_size=0.3, random_state=1234)

    st = StandardScaler()
    X_y_train = pd.concat([X_tra, y_tra], axis=1).reset_index()
    X_y_train.iloc[:, 1:-1] = st.fit_transform(X_y_train.iloc[:, 1:-1])
    X_y_test = pd.concat([X_te, y_te], axis=1).reset_index()
    X_y_test.iloc[:, 1:-1] = st.fit_transform(X_y_test.iloc[:, 1:-1])
    X_tra = X_y_train.iloc[:, 1:-1]
    y_tra = X_y_train.iloc[:, -1]
    X_te = X_y_test.iloc[:, 1:-1]
    y_te = X_y_test.iloc[:, -1]
    X_tra = pd.DataFrame(st.fit_transform(X_tra))
    X_te = pd.DataFrame(st.transform(X_te))

    cost_fn = 42
    tuple = (0, 1)

    print('DDAE')
    # ddae(X_tra, y_tra, X_te, y_te, lb, tuple)
    print('IML')
    # iml(X_tra, y_tra, X_te, y_te, lb, tuple)

    y_tra = y_tra.map({tuple[0]: 0, tuple[1]: 1})
    y_te = y_te.map({tuple[0]: 0, tuple[1]: 1})

    print('RUSBoost')
    rusboost(X_tra, y_tra, X_te, y_te)
    print('MWMOTE')
    mwmote(X_tra, y_tra, X_te, y_te)
    print('SMOTE')
    smote(X_tra, y_tra, X_te, y_te)
    print('MetaCost')
    metacost(X_tra, y_tra, X_te, y_te, cost_fn, lb)
    print('AdaBoost')
    adaBoost(X_tra, y_tra, X_te, y_te)
    print('cost-sensitive decision tree')
    costsensitivedct(X_tra, y_tra, X_te, y_te, cost_fn)
    print('CAdaMEC')
    cadamec(X_tra, y_tra, X_te, y_te, cost_fn, n_estimators=51)
    print('SPE')
    self_paced(X_tra, y_tra, X_te, y_te)
