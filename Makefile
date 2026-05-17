.PHONY : clean install airflow-up airflow-down

clean:
	@for file in ".ipynb_checkpoints" "*.egg-info" "__pycache__"; do \
	find -name  $$file | xargs rm -fr; \
	done;

install:
	@uv sync

airflow-up:
	docker compose -f compose.yml -f compose.airflow.yml up --build

airflow-down:
	docker compose -f compose.yml -f compose.airflow.yml down

        
