---
---

# Making a package

### The `__init__.py`
- A python package: needs a `__init__.py` file.
	- It tells python that the directory contains a package
	- Always required
- The code inside this file is gets run whenever the package is imported
	- It could include some default imports

### PIP
- `pip` is a Python package manager that helps with installing and uninstalling Python packages.

### PIP installing
- A `setup.py` file is required.
	- It should exists at the same level of the package folder.
	- The file contains metadata about the package:

- `setup.py` content:
```python
from setuptools import setup

setup(
	name = 'myPackage',
	version = '0.1',
	description = 'Perros package',
	packages = ['myPackage'],
	zip_safe = False
)
```

##### Local installing
1. Go to the parent directory containing the package folder and the `setup.py` file
2. Run the following
	- `pip insall .`
3. The package is installed in the directory defined by PIP.


## Putting code in PyPi

- First upload the package to the test repository.
- Then on the regular repository
- Go to the package directory
- Add the following two files:
	1. `license.txt`
	2. `README.md`
	3. `setup.cfg`
	4. `setup.py`
- Every package needs a unique name
- Run: `python setup.py sdist`
	- This will create a


```shell
cd binomial_package_files
python setup.py sdist
pip install twine

# commands to upload to the pypi test repository
twine upload --repository-url https://test.pypi.org/legacy/ dist/*
pip install --index-url https://test.pypi.org/simple/ dsnd-probability

# command to upload to the pypi repository
twine upload dist/*
pip install dsnd-probability
```