import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm
from sklearn.model_selection import StratifiedKFold
from copy import deepcopy


def feature_importance(X_train, y_train, X_test, y_test, cols):
    y_train, y_test = y_train.astype(int), y_test.astype(int)
    X_train_base = X_train[cols]
    X_test_base = X_test[cols]
    model = LinearDiscriminantAnalysis()
    model.fit(X_train_base.to_numpy(), y_train)
    base_model = deepcopy(model)
    base_model.fit(X_train_base.to_numpy(), y_train)
    skf = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)
    base_f1 = f1_score(y_test, model.predict(X_test_base))
    # Feature importance
    score_df = {'feature': [], 'drop_importance': [],
                'perm_importance': [],
                # 'single_feature':[]
                }
    for feat in tqdm(cols):
        score_df['feature'].append(feat)
        selected_features = np.setdiff1d(cols, feat)
        X_train_tmp = X_train[selected_features].to_numpy()
        model.fit(X_train_tmp, y_train)
        f1_list = []
        for train_index, test_index in skf.split(X_test, y_test):
            X_test_fold = X_test_base.iloc[test_index]
            y_test_fold = y_test.iloc[test_index]
            X_test_tmp = X_test_fold[selected_features].to_numpy()
            pred = model.predict(X_test_tmp)
            f1_list.append(base_f1 - f1_score(y_test_fold, pred))
        score_df['drop_importance'].append(np.mean(f1_list))

        f1_list = []
        for train_index, test_index in skf.split(X_test, y_test):
            X_test_fold = X_test_base.iloc[test_index]
            y_test_fold = y_test.iloc[test_index]
            X_test_tmp = X_test_fold.copy()
            X_test_tmp[feat] = np.random.permutation(X_test_tmp[feat])
            X_test_tmp = X_test_tmp.to_numpy()
            pred = base_model.predict(X_test_tmp)
            f1_list.append(base_f1 - f1_score(y_test_fold, pred))
        score_df['perm_importance'].append(np.mean(f1_list))

        # selected_features = [feat]
        # X_train_tmp = X_train[selected_features].to_numpy()
        # model.fit(X_train_tmp, y_train)
        # f1_list = []
        # for train_index, test_index in skf.split(X_test_base,y_test):
        #   X_test_fold = X_test.iloc[train_index]
        #   y_test_fold = y_test.iloc[train_index]
        #   X_test_tmp = X_test_fold[selected_features].to_numpy()
        #   pred = model.predict(X_test_tmp)
        #   f1_list.append(base_f1 - f1_score(y_test_fold, pred))
        # score_df['single_feature'].append(np.mean(f1_list))

    score_df = pd.DataFrame(score_df)
    score_df = score_df.set_index('feature')
    score_df['score'] = score_df['perm_importance']
    score_df = score_df.drop('drop_importance', 1)
    score_df = score_df.drop('perm_importance', 1)
    score_df.plot(kind='bar', title='Feature importances');

    res_dict = {'q': [], 'f1': []}
    model = ExtraTreesClassifier(random_state=0)
    for x in tqdm(np.unique(score_df['score'])):
        res_dict['q'].append(x)
        selected_features = score_df[score_df['score'] >= x].index
        X_train_tmp = X_train[selected_features].to_numpy()
        model.fit(X_train_tmp, y_train)

        f1_list = []
        for train_index, test_index in skf.split(X_test, y_test):
            X_test_fold = X_test.iloc[test_index]
            y_test_fold = y_test.iloc[test_index]
            X_test_tmp = X_test[selected_features].to_numpy()
            pred = model.predict(X_test_tmp)
            f1_list.append(f1_score(y_test, pred))
        res_dict['f1'].append(np.median(f1_list))

    res_df = pd.DataFrame(res_dict)
    display(res_df)
    max = res_df.loc[:, 'f1'].max()
    best_threshold = res_df[res_df['f1'] == max].iloc[-1]['q']
    new_cols = score_df[score_df['score'] >= best_threshold].index
    return score_df, res_df, new_cols
