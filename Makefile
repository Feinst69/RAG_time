.PHONY : clean, install, eval

clean:
	@for file in ".ipynb_checkpoints" "*.egg-info" "__pycache__"; do \
	find -name  $$file | xargs rm -fr; \
	done;

install:
	@uv sync
        
eval:
	@uv run src/rag_time/eval/rag_eval.py