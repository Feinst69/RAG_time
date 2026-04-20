.PHONY : clean, install

clean:
	@for file in ".ipynb_checkpoints" "*.egg-info" "__pycache__"; do \
	find -name  $$file | xargs rm -fr; \
	done;

install:
	@uv sync

        