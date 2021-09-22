import numpy as np  # Math library
import pandas as pd  # To work with dataset
from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python_operator import PythonOperator
import os
from daghelper.main import target_variable_dist_plot, credit_amount_plotting, feature_engineering, delete_variables, create_plot_comparison, random_forest_summary, gaussian_summary, house_owning_plot, saving_accounts_plot,  corr_plot, write_dataframe

# CONSTANTS
AIRFLOW_HOME = os.getenv('AIRFLOW_HOME')

default_args = {
    'owner': 'Yovani Santiago',
    'start_date': days_ago(1)
}


def read_credit_risk_data(**context):
    # Importing the data
    df_credit = pd.read_csv(
        f"{AIRFLOW_HOME}/dags/input/german_credit_data.csv", index_col=0)
    print(df_credit.head(10))
    task_instance = context['task_instance']
    task_instance.xcom_push(key="credit_risk", value='read')


def data_explore(**kwargs):
    df_credit = pd.read_csv(
        f"{AIRFLOW_HOME}/dags/input/german_credit_data.csv", index_col=0)
    print("showing shape of data")
    print(df_credit.info())
    print("showing unique data")
    print(df_credit.nunique())
    print("Showing head data")
    print(df_credit.head())


def variables_remover_task(**context):
    enhanced_df = delete_variables()
    print(enhanced_df.head(10))
    task_instance = context['task_instance']
    task_instance.xcom_push(key="variables_removed", value='True')


# Define DAG
with DAG(
        'kernel-migration',
        default_args=default_args,
        schedule_interval=None,
) as dag:
    read_data = PythonOperator(
        task_id='read_credit_risk_data',
        python_callable=read_credit_risk_data,
    )
    explore_data = PythonOperator(
        task_id='Explore_data',
        python_callable=data_explore

    )

    plot_credit_dist = PythonOperator(
        task_id='create_distribution_plot',
        python_callable=target_variable_dist_plot

    )

    plot_house_owning = PythonOperator(
        task_id='housing_owning_risk_plots',
        python_callable=house_owning_plot

    )

    plot_credit_amount = PythonOperator(
        task_id='credit_amount_plotting',
        python_callable=credit_amount_plotting

    )

    plot_sav_acct = PythonOperator(
        task_id='saving_account_plotting',
        python_callable=saving_accounts_plot

    )

    feat_eng = PythonOperator(
        task_id='feature_engineering',
        python_callable=feature_engineering

    )

    del_var = PythonOperator(
        task_id='remove_variables',
        python_callable=variables_remover_task

    )

    correlation_plot = PythonOperator(
        task_id='correlation_plotting',
        python_callable=corr_plot

    )

    prep_data = PythonOperator(
        task_id='preprocess_data',
        python_callable=create_plot_comparison

    )

    model_1 = PythonOperator(
        task_id='model_1_random_forest',
        python_callable=random_forest_summary

    )

    model_2 = PythonOperator(
        task_id='model_2_guassian_model',
        python_callable=gaussian_summary

    )

    write_data = PythonOperator(
        task_id='write_data',
        python_callable=write_dataframe

    )


read_data >> explore_data >> [plot_credit_dist, plot_house_owning, plot_credit_amount, plot_sav_acct] >> feat_eng  # Defining the task dependencies

feat_eng >> del_var >> correlation_plot >> prep_data >> [model_1, model_2]

model_1 >> write_data

model_2 >> write_data
