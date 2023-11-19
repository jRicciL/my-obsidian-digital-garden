---
---

## Documentation
- Additional text or illustrations that comes with or is embedded in software code.
- Used to:
	- Clarify complex parts of code
	- NAvigate code easily
	- Describe use and purpose of components

#### Types of Documentation

##### Line level
- `# In line comments`
- To many comments is a signal of bad code => Refactoring is needed
- Comments are valuable to explain what code cannot
	- History behind why a function has certain parameters that could seen arbitrary but they are not

#Udacity-questions-AWS
1. ✅ Comments are useful for clarifying complex code
2. ❌ ~~You never have too many comments~~
3. ❌ ~~Comments are only for unreadable parts of code~~
4. ✅ Readable code is preferable over having comments to make your code readable.

##### Function or module level
#Docstrings => ==Docstrings==
- All functions should have `docstrings`
- Should include a single line summary
- Should include an explanation of `arguments`
- Should include an explanation of `returns`
- Should include the type of each *Argument* and *returned* value

#Udacity-questions-AWS
*Which of the following statements about docstrings are true?*
- ❌ ~~Multiline dockstrings are better than single line docstrings.~~
- ✅ Docstrings explain the purpose of a function or module
- ❌ ~~Docstrings and comments are interchangeable~~
- ✅ You can add whatever details you want in a docstring
- ❌ ~~Not including a docstring will cause an error~~

##### Project documentation
- `README.md` file is a great first step in project documentation.
	- Should explain what the package does.
	- Lists dependencies
	- Provide sufficiently detailed instructions on how to use it.

