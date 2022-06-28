all: clean setup install

clean:
	$(RM) -r ymir.egg-info/ dist/ build/

setup:
	python3 setup.py sdist bdist_wheel

install:
	pip3 install .
