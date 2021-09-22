# to split the data
from sklearn.model_selection import train_test_split, KFold, cross_val_score
# To evaluate our model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, fbeta_score

from sklearn.model_selection import GridSearchCV

# Algorithmns models to be compared
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
import numpy as np
import pandas as pd

#from sklearn.utils import resample
from sklearn.metrics import roc_curve


def create_variables(df_credit):
    # Let's look the Credit Amount column
    interval = (18, 25, 35, 60, 120)

    cats = ['Student', 'Young', 'Adult', 'Senior']
    df_credit["Age_cat"] = pd.cut(df_credit.Age, interval, labels=cats)

    #df_good = df_credit[df_credit["Risk"] == 'good']
    #df_bad = df_credit[df_credit["Risk"] == 'bad']

    df_credit['Saving accounts'] = df_credit['Saving accounts'].fillna(
        'no_inf')
    df_credit['Checking account'] = df_credit['Checking account'].fillna(
        'no_inf')

    # Purpose to Dummies Variable
    df_credit = df_credit.merge(pd.get_dummies(
        df_credit.Purpose, drop_first=True, prefix='Purpose'), left_index=True, right_index=True)
    # Sex feature in dummies
    df_credit = df_credit.merge(pd.get_dummies(
        df_credit.Sex, drop_first=True, prefix='Sex'), left_index=True, right_index=True)
    # Housing get dummies
    df_credit = df_credit.merge(pd.get_dummies(
        df_credit.Housing, drop_first=True, prefix='Housing'), left_index=True, right_index=True)
    # Housing get Saving Accounts
    df_credit = df_credit.merge(pd.get_dummies(
        df_credit["Saving accounts"], drop_first=True, prefix='Savings'), left_index=True, right_index=True)
    # Housing get Risk
    df_credit = df_credit.merge(pd.get_dummies(
        df_credit.Risk, prefix='Risk'), left_index=True, right_index=True)
    # Housing get Checking Account
    df_credit = df_credit.merge(pd.get_dummies(
        df_credit["Checking account"], drop_first=True, prefix='Check'), left_index=True, right_index=True)
    # Housing get Age categorical
    df_credit = df_credit.merge(pd.get_dummies(
        df_credit["Age_cat"], drop_first=True, prefix='Age_cat'), left_index=True, right_index=True)

    return df_credit


def remove_variables(df_credit):
    # Excluding the missing columns
    print("Removing saving accounts column")
    del df_credit["Saving accounts"]
    print("Removing Checking account column")
    del df_credit["Checking account"]
    print("Removing Purpose column")
    del df_credit["Purpose"]
    print("Removing Sex column")
    del df_credit["Sex"]
    print("Removing Housing column")
    del df_credit["Housing"]
    print("Removing Age cat column")
    del df_credit["Age_cat"]
    print("Removing Risk column")
    del df_credit["Risk"]
    print("Removing Risk good column")
    del df_credit['Risk_good']
    return df_credit


def split_data(df_credit):
    df_credit['Credit amount'] = np.log(df_credit['Credit amount'])

    # Creating the X and y variables
    X = df_credit.drop('Risk_bad', 1).values
    y = df_credit["Risk_bad"].values

    # Spliting X and y into train and test version
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)

    return [X_train, X_test, y_train, y_test]


def one_hot_encoder(df, nan_as_category=False):
    original_columns = list(df.columns)
    categorical_columns = [
        col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns,
                        dummy_na=nan_as_category, drop_first=True)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


def model_eval(X_train, y_train):

    # to feed the random state
    seed = 7

    # prepare models
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('RF', RandomForestClassifier()))
    models.append(('SVM', SVC(gamma='auto')))
    models.append(('XGB', XGBClassifier()))

    # evaluate each model in turn
    results = []
    names = []
    scoring = 'recall'

    for name, model in models:
        kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
        cv_results = cross_val_score(
            model, X_train, y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

    return [results, names]


def random_forest_check(X_train, X_test, y_train, y_test):
    # Seting the Hyper Parameters
    param_grid = {"max_depth": [3, 5, 7, 10, None],
                  "n_estimators": [3, 5, 10, 25, 50, 150],
                  "max_features": [4, 7, 15, 20]}

    # Creating the classifier
    model = RandomForestClassifier(random_state=2)

    grid_search = GridSearchCV(
        model, param_grid=param_grid, cv=5, scoring='recall', verbose=4)
    grid_search.fit(X_train, y_train)

    print(grid_search.best_score_)
    print(grid_search.best_params_)

    rf = RandomForestClassifier(
        max_depth=None, max_features=10, n_estimators=15, random_state=2)

    # trainning with the best params
    rf.fit(X_train, y_train)
    # Testing the model
    # Predicting using our  model
    y_pred = rf.predict(X_test)

    # Verificaar os resultados obtidos
    print(accuracy_score(y_test, y_pred))
    print("\n")
    print(confusion_matrix(y_test, y_pred))
    print("\n")
    print(fbeta_score(y_test, y_pred, beta=2))


def gaussian_check(X_train, X_test, y_train, y_test):
    # Criando o classificador logreg
    GNB = GaussianNB()

    # Fitting with train data
    model = GNB.fit(X_train, y_train)

    # Printing the Training Score
    print("Training score data: ")
    print(model.score(X_train, y_train))

    y_pred = model.predict(X_test)

    print(accuracy_score(y_test, y_pred))
    print("\n")
    print(confusion_matrix(y_test, y_pred))
    print("\n")
    print(classification_report(y_test, y_pred))
    # Predicting proba
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # Generate ROC curve values: fpr, tpr, thresholds
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

    return [fpr, tpr]
