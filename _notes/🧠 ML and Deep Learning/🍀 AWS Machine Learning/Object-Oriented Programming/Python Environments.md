---
---

# Python environments

A virtual environment is a silo-ed Python installation apart from the main Python installation.
- This way we can install packages and delete the virtual environment without affecting the main Python installation

### Conda vs venv
#conda vs #venv

#### Conda
`conda` does two things:
1. Manages packages:
	- Python packages but also packages from other languages/sources
		- Agnostic package manager
		- 
2. Manages environments

- It makes it Easy to install python packages.
- It allows to create python environmentes.
- It was invented because `pip` could not handle data science packages outside python.

##### Use
```python
conda create --name environmentName
source activate environmentName
conda install numpy
```

#### `pip` and `Venv`
- `Venv` is a python environment manager that comes preinstalled with python 3.
	- It only manages environments
- `pip` only manages packages:
	- Only manages python packages

##### Use
```python
python3 -m venv environmentName
source environmentName/bin/activate
pip install numpy
```
- Creates a new folder with the python installation

## Which to choose
- `Conda` is very helpful for data science projects
	- <mark style='background-color: #FFA793 !important'>NOTE:</mark> It is necessary to install `pip` inside the `conda` environment to install new packages using `pip` only inside teh environment scope.
	
### venv
