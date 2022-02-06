---
---

# Lesson 4:
## Clean and modular code

- ==Production code==:
	- Runs on production servers
	- Ensure reliability and efficiency before to be public
		- *Clean code* => 
			- Readable
			- Simple
			- Concise
		- *Modular* =>
			- Program is logically broken up:
				- Functions and modules
				- Improves organization, efficiency, and reusability
			- Also it is easy to:
				- Reuse the code
				- Write less code 
				- Read the code
				- Collaborate on code
		- *Module* => Afile
			- modules allow code to be reused by encapsulating them into files that can be imported into other files.

### Tips
- ==DRY==: *Dont repeat yourself*
- Use meaningful names
- Be descriptive and imply type
	- With booleans use `is_` or `has_`
	- Use verbs for `functions`
	- Use nouns for `varaibles`
- Be consistent but clearly differentiate
- Use the `math` module, it is great!!
- Avoid abbreviations and single letters

## Refactoring
- Messy and repetitive code => 
	- Go back to refactoring after a working code
	- ==Refactoring==:
		- Restructuring code to improve internal structure without changing external functionality.
	
## Writing Modular Code
- DRY => Don't repeat yourself
- 🤢 ==Spagetti code== -> It is common among data scientists 
	- Avoid it!!

#### Tips
- `Comprhension` instead of `for loops`
- Generalize common tasks into one function
- Abstract the logic to improve readability
- Minimize the number of entities
	- `Functions`, `classes`, `modules`
- Make sure that each function does just one thing
	- Avoid unnecessary side effects
- Arbitrary variable names can be more effective in certain circumstances
	- For `general functions`
- Avoid using more than three parameters per function
	- This does not always apply

## Efficient Code
- Two parts to make the code efficient:
	- Execution time
	- Memory

- Code optimization is *context* dependent => Some task require more efficiency than others.

#### Tips
- Use vector operations over foor loops

