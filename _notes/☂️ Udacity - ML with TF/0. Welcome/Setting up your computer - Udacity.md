---
---

# Setting up your computer

- A ==package== is a bunch of ==modules== => each module consists of a set of classes and function definitions.
- **Package managers** are used to installing `libraries` and other software.
- `pip` is the default package manager for Python libraries.
- `conda` is usually more focused on data science while `pip` is for more general use
- ==python environment== => 
	- Python interpreter
	- Python packages
	- Utility scripts
- **Environments** are used for specific proposes or projects
- Export the list of packages in an environment file =>
	-  `pop freeze > requirements.txt`
- Things installed when install *Miniconda*:
	- *Python*, *Conda*, and its dependencies

## Jupyter notebooks
- **Jupyter** comes from the combination of **Ju**lia, **Pyt**hon, and **R**.
- The notebook is a web application that allows you to combine explanatory text, math equations, code, and visualizations.

**Used for:**
- *Data cleaning and exploration*
- *Visualization*
- *Machine Learning*
- *Big Data analysis*

### Notebooks are *Literate Programming*
Notebooks are a form of ==literate programming== proposed by Donald Knuth in 1984.
- The documentation is written as a narrative alongside the code
- This idea has been extended to the `Eve` programming language

### How notebooks work
![[Pasted image 20211220180356.png]]

- The `notebook` is rendered as a web app
- The `kernel` runs the code and sends it back to the server
- The `server` sends the information to the browser to render the results