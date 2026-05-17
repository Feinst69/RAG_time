FROM apache/airflow:2.9.0-python3.12

# Passer en utilisateur root si on a besoin d'installer des paquets systèmes
USER root
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
         build-essential \
         curl \
  && apt-get autoremove -yqq --purge \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

USER airflow

# Copier et installer les dépendances du projet
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
