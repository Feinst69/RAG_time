from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'rag_ingestion_pipeline',
    default_args=default_args,
    description='Pipeline for embedding and uploading RAG data',
    schedule_interval='@daily',
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['rag', 'ingestion'],
) as dag:
    
    # Étape 1 : Générer les embeddings avec le script small
    # On spécifie explicitement le chemin complet dans le container
    embed_task = BashOperator(
        task_id='generate_embeddings',
        bash_command='python /opt/airflow/scripts/embed_data_small.py -i /opt/airflow/data/aa_dataset-tickets-multi-lang-5-2-50-version.csv -o /opt/airflow/data/output.jsonl'
    )

    # Étape 2 : Uploader vers Qdrant
    upload_task = BashOperator(
        task_id='upload_to_qdrant',
        bash_command='python /opt/airflow/scripts/qdrant_upload.py -i /opt/airflow/data/output.jsonl'
    )

    # Définition de l'ordre d'exécution
    embed_task >> upload_task
