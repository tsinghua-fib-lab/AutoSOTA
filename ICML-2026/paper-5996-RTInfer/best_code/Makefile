.PHONY: install install-dev test rebuttal figures case docker-build clean

install:
	python3 -m pip install -e .

install-dev:
	python3 -m pip install -e ".[dev]"

test:
	python3 -m pytest

rebuttal:
	./rebuttal_experiments/run_all_rebuttal.sh

figures:
	python3 paper_figures/make_revised_figures.py

case:
	./case_studies/jetson_nano_case/run_case.sh

docker-build:
	docker build -t rtinfer-reproduction .

clean:
	rm -rf build dist *.egg-info .pytest_cache
