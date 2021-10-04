all: clean setup install

clean:
	$(RM) -r pox3.egg-info/ dist/ build/

setup:
	python3 setup.py sdist bdist_wheel

install:
	pip install .