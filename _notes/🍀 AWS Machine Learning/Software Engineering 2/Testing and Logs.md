---
---

# Testing and Logs

## Testing
- Testing is essential before deployment.
- It helps catch error and faulty conclusions before make any major impact.
- It is not a common practice among data scientists.

### Testing and Data Science

- Proper testing is necessary to avoid unexpected surprises and have confidence in the results.
- 

==Test Driven development== -> #TDD:
- Writing tests before developing code to implement tasks

==Unit Test== -> #UnitTest
- Test that covers a small unit of code, usually a function.

### Unit Test

- Test in a way that is repeatable and automated.
- Ideally we want to run all tests and cleanly showing the results.

##### Advantages and disadvantages

**Advantages**
- They are isolated from the rest of the program.
- No dependencies are involved.
- They don't require access to databases, APIs, or other external sources

**Disadvantages**
- They are not always enough to prove that the program is working as expected.
- Because they are isolated => We cannot test the communication and transferring data between isolated functions 
	- Here, we will need to use integration tests

#### Unit Testing Tools
##### pytest -> #pytest

1. Write your function in a file => `nearest_sq.py`

```python
def nearest_square(num):
	"""Return the nearest perfect square that is less
	than or eaqual to `num`"""
	root = 0
	while (root + 1) ** 2 <= num:
		root += 1
	return root ** 2
```

2. Create a `test_` file, calling your function => `test_nearest_sq.py`:

```python
from nearest import nearest_square

def test_nearest_square_5():
	assert( nearest_square(5) == 4)
	
# This is wrong
def test_nearest_square_9():
	assert( nearest_square(9) == 5)
	
def test_nearest_square_n12():
	assert( nearest_square(-12) == 0)
```

3. Run `pytest` inside the `testing` directory, this will execute all test files
4. ✅   In the test output, ==periods== represent successful  unit tests.
	- `test_file.py ...`
5. ❌ ==Fs== represent failed unit tests.
	- `test_file.py .F.`
6. The best practice is to have ==only one `assert`== per test

## Test-driven Development 
#TDD
- *Test-driven development* ===>== Writing tests before write the code being tested.
	- The test fails at firsts
	- A task is implemented when it pass the test
- Tests can check for different scenarios and edge cases before event start write the function.
	- The test can be implemented to get immediate feedback on whether it works or not.
- When refactoring or adding to the code:
	- Help to assure that the rest of the code doesn't have been break.
	- Allows ensure that the function behavior is repeatable, regardless of external parameters.

![[Captura de Pantalla 2021-09-08 a la(s) 7.44.25.png]]

## Logging

Logging is the process of recording messages to describe events that have occurred while running the software.

#### Tips
##### Be professional and clear
```python
Bad: This is not working
Good: Could't parse the file
```

##### Be concise and use normal capitalization
```
Bad: Start Product Recommendation Process
Bad: We have completed the steps necessary and will now proceed with the recommendation process for the records in our product database.

Good: Generating product recommendations.
```

##### Choose the appropriate level for logging
- 🟠  *Debug* => use this level for anything that happens in the program
- 🔴  *Error* => Use this level to record any error that occurs
- 🟢  *Info* => Use this level to record all actions that are user driven or system specific, such as regularly scheduled operations.

<mark style='background-color: #FFA793 !important'>ERROR</mark> is the appropriate level for this error message, though more information on where, when, and how this occurred would be useful for debugging. It's best practice to use concise and clear language that is professional and uses normal capitalization. This way, the message is efficient and easily understandable. The second sentence seems quite unclear and personal, so we should remove that and communicate it elsewhere.

##### Provide any useful information
```
Bad: Failed to read location data
Good: Failed to read location data: store_id 8324971
```

