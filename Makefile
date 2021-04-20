.PHONY: tests clean clean-pyc upload-pypi-test upload-pypi requirements docs \
	code-cov

clean:
	python setup.py clean

clean-pyc:
	find . -name '*.pyc' -exec rm {} \;

upload-pypi-test:
	python setup.py sdist bdist_wheel && \
	  twine upload --repository testpypi dist/* && \
	  rm -rf dist

upload-pypi:
	python setup.py sdist bdist_wheel && \
	  twine upload --repository pypi dist/* && \
	  rm -rf dist

requirements:
	pip install -r requirements-dev.txt

docs:
	make clean && \
    rm -rf docs/build/ && \
	rm -rf docs/source/api/ && \
	  sphinx-autogen docs/source/*.rst && \
	  python -m sphinx -E "docs/source" "docs/build" -W

code-cov:
	pytest && \
	mypy piper/
