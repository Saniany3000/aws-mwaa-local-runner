import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# it's a library that we work with plotly
import plotly.graph_objs as go  # it's like "plt" of matplot
from plotly import tools
from collections import Counter  # To do counter of some features
from collections import Counter  # To do counter of some features
import os
# from sklearn.model_selection import train_test_split, KFold, cross_val_score # to split the data

from daghelper.models import create_variables, split_data, model_eval, random_forest_check, gaussian_check, one_hot_encoder, remove_variables

AIRFLOW_HOME = os.getenv('AIRFLOW_HOME')


# Importing the data
CREDIT_DATAFRAME = pd.read_csv(
    f"{AIRFLOW_HOME}/dags/input/german_credit_data.csv", index_col=0)

INTERVAL = (18, 25, 35, 60, 120)
CATEGORIES = ['Student', 'Young', 'Adult', 'Senior']
CREDIT_DATAFRAME["Age_cat"] = pd.cut(
    CREDIT_DATAFRAME.Age, INTERVAL, labels=CATEGORIES)

DF_GOOD = CREDIT_DATAFRAME[CREDIT_DATAFRAME["Risk"] == 'good']
DF_BAD = CREDIT_DATAFRAME[CREDIT_DATAFRAME["Risk"] == 'bad']


# Path Helper

def check_path(path):
    if not os.path.exists(f"{AIRFLOW_HOME}/dags/{path}"):
        os.mkdir(f"{AIRFLOW_HOME}/dags/{path}")
        print(f"path {AIRFLOW_HOME}/dags/{path} sucessfully created")
    else:
        print(f"path {AIRFLOW_HOME}/dags/{path} exists")


#  PLOTTING FUNCTIONS
#
def save_plot(path: str, fig: go, file_name: str) -> go:
    check_path(path)
    fig.write_image(f"{AIRFLOW_HOME}/dags/{path}/{file_name}")
    print(f"image saved as {AIRFLOW_HOME}/dags/{path}/{file_name}")


def target_variable_dist_plot():
    df_credit = CREDIT_DATAFRAME
    trace0 = go.Bar(
        x=df_credit[df_credit["Risk"] ==
                    'good']["Risk"].value_counts().index.values,
        y=df_credit[df_credit["Risk"] == 'good']["Risk"].value_counts().values,
        name='Good credit'
    )

    trace1 = go.Bar(
        x=df_credit[df_credit["Risk"] ==
                    'bad']["Risk"].value_counts().index.values,
        y=df_credit[df_credit["Risk"] == 'bad']["Risk"].value_counts().values,
        name='Bad credit'
    )

    data = [trace0, trace1]

    layout = go.Layout(

    )

    layout = go.Layout(
        yaxis=dict(
            title='Count'
        ),
        xaxis=dict(
            title='Risk Variable'
        ),
        title='Target variable distribution'
    )

    fig = go.Figure(data=data, layout=layout)

    print("target variable distribution plot created, saving")

    save_plot("output/plots", fig, "target_variable_distribution.png")


def credit_amount_plotting():
    df_good = DF_GOOD
    df_bad = DF_BAD
    trace0 = go.Box(
        y=df_good["Credit amount"],
        x=df_good["Age_cat"],
        name='Good credit',
        marker=dict(
            color='#3D9970'
        )
    )

    trace1 = go.Box(
        y=df_bad['Credit amount'],
        x=df_bad['Age_cat'],
        name='Bad credit',
        marker=dict(
            color='#FF4136'
        )
    )

    data = [trace0, trace1]

    layout = go.Layout(
        yaxis=dict(
            title='Credit Amount (US Dollar)',
            zeroline=False
        ),
        xaxis=dict(
            title='Age Categorical'
        ),
        boxmode='group'
    )
    fig = go.Figure(data=data, layout=layout)

    print("credit amount plot created, saving")

    save_plot("output/plots", fig, "credit_amount_plot.png")


def house_owning_plot():
    df_credit = CREDIT_DATAFRAME
    # First plot
    trace0 = go.Bar(
        x=df_credit[df_credit["Risk"] ==
                    'good']["Housing"].value_counts().index.values,
        y=df_credit[df_credit["Risk"] ==
                    'good']["Housing"].value_counts().values,
        name='Good credit'
    )

    # Second plot
    trace1 = go.Bar(
        x=df_credit[df_credit["Risk"] ==
                    'bad']["Housing"].value_counts().index.values,
        y=df_credit[df_credit["Risk"] ==
                    'bad']["Housing"].value_counts().values,
        name="Bad Credit"
    )

    data = [trace0, trace1]

    layout = go.Layout(
        title='Housing Distribuition'
    )

    fig = go.Figure(data=data, layout=layout)

    print("credit amount plot created, saving")

    save_plot("output/plots", fig, "house_owning_risk.png")


def saving_accounts_plot():
    df_good = DF_GOOD
    df_bad = DF_BAD
    count_good = go.Bar(
        x=df_good["Saving accounts"].value_counts().index.values,
        y=df_good["Saving accounts"].value_counts().values,
        name='Good credit'
    )
    count_bad = go.Bar(
        x=df_bad["Saving accounts"].value_counts().index.values,
        y=df_bad["Saving accounts"].value_counts().values,
        name='Bad credit'
    )

    box_1 = go.Box(
        x=df_good["Saving accounts"],
        y=df_good["Credit amount"],
        name='Good credit'
    )
    box_2 = go.Box(
        x=df_bad["Saving accounts"],
        y=df_bad["Credit amount"],
        name='Bad credit'
    )

    scat_1 = go.Box(
        x=df_good["Saving accounts"],
        y=df_good["Age"],
        name='Good credit'
    )
    scat_2 = go.Box(
        x=df_bad["Saving accounts"],
        y=df_bad["Age"],
        name='Bad credit'
    )

    #data = [scat_1, scat_2, box_1, box_2, count_good, count_bad]

    fig = tools.make_subplots(rows=2, cols=2, specs=[[{}, {}], [{'colspan': 2}, None]],
                              subplot_titles=('Count Saving Accounts', 'Credit Amount by Savings Acc',
                                              'Age by Saving accounts'))

    fig.append_trace(count_good, 1, 1)
    fig.append_trace(count_bad, 1, 1)

    fig.append_trace(box_2, 1, 2)
    fig.append_trace(box_1, 1, 2)

    fig.append_trace(scat_1, 2, 1)
    fig.append_trace(scat_2, 2, 1)

    fig['layout'].update(height=700, width=800,
                         title='Saving Accounts Exploration', boxmode='group')
    print("saving accounts plot created, saving")

    save_plot("output/plots", fig, "saving_account_plot.png")


def feature_engineering():
    one_hot_encoder(CREDIT_DATAFRAME)

    print("feature engineering simulated, features created")


def delete_variables():

    #  CREATE VARIABLES
    enhanced_df = create_variables(CREDIT_DATAFRAME)
    print("removing variables from dataframe")
    final_df = remove_variables(enhanced_df)
    return final_df

#  ALGORYTHM COMPARSION


model_df = delete_variables()


def corr_plot():
    print("creating with added variables  correlation plot")
    plt.figure(figsize=(14, 12))
    sns.heatmap(model_df.astype(float).corr(), linewidths=0.1, vmax=1.0,
                square=True,  linecolor='white', annot=True)
    print("saving")
    plt.savefig(f"{AIRFLOW_HOME}/dags/output/plots/correlation_plot.png")


# SPLIT DATA
X_train, X_test, y_train, y_test = split_data(model_df)

# RUN MODELS


def create_plot_comparison():

    print("Evaluation models")
    results, names = model_eval(X_train, y_train)
    print("Models evaluation completed, plotting")
    print(names)
    # boxplot algorithm comparison
    fig = plt.figure(figsize=(11, 6))
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    print("Finished plotting, saving image")
    plt.savefig(f"{AIRFLOW_HOME}/dags/output/plots/algorithm_comparison.png")


# MODEL 1
def random_forest_summary():
    random_forest_check(X_train, X_test, y_train, y_test)

# MODEL 2


def gaussian_summary():
    fpr, tpr = gaussian_check(X_train, X_test, y_train, y_test)
    # Plot ROC curve
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    print("Saving  gaussian plot")
    plt.savefig(f"{AIRFLOW_HOME}/dags/output/plots/gaussian_roc_curve.png")


def write_dataframe():
    print("writing_data in the output folder")
    check_path('output/dataframes')
    model_df.to_csv(
        f"{AIRFLOW_HOME}/dags/output/dataframes/output_tagged_risk_data.csv", index=False, encoding='utf-8')
    print(
        f"succesfully wrote data in as '{AIRFLOW_HOME}/dags/output/dataframes/output_tagged_risk_data.csv' ")
