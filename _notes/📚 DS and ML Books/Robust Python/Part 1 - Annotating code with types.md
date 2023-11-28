# C1 - Introduction to Robust Python

## Book’s intention
- The book is about making Python code more:
	- **Robust**
	- **Manageable**
	- **Healthy**
	- **Clean and maintainable**
		- Clean: Organized, documented, verbose (just enough), better naming patterns, functions should be short and simple
	- Capable of **performing** without **failure**
	- **Communication** intended:
		- Easy to read → Cheap to consume
- The zen of python:
	→ *There should be one - and preferably only one - obvious way to do it*
- The law of the least surprise:
	- *A program should always respond to the user in the way that astonishes them the least.*

## Examples of intent in Python
### Collections
- **==List==**: Mutable, duplicated elements. Used to iterate over.
- **==String==**: Immutable collection of characters.
- **==Generator==**: A collection to be iterated over. Never indexed into. Great for for computationally expensive or infinite collections.
- **==Tuple==**: Immutable. Well for indexing elements. Rarelly iterated over.
- **==Set==**: No duplicates. No ordered. Mutable.
- ==**Dictionary**==: A mapping from keys to values. Typically iterated over. Indexed using dynamic keys.
- `frozenset`:  An immutable set
- `OrdererDict`: A dictionary with ordered elements based on insertion time.
- `defaultdict`: Dict that provides a default value if key is missing. From `collections.defaultdict`
- `Counter`: Special use case for a dictionary that counts how many times an element appears:
```python
from collections import Counter
def create_count_mappint(cookbooks: list[Coockbook]):
	return Counter(book.author for book in cookbooks)
```

### Iteration
The type of iteration you choose dictates the intent you convey.
- **==for loops==**: Used for iterating over elements in a collection or range. Perform a side effect.
- ==**while loops**==: iterate as long as a certain condition is true.
- ==**Comprehensions**==: Transforming one collection into another. It does not have side effects.
- ==**Recursions**==: used when the substructure of a collection is identical to the structure of a collection

# C2 - Introduction to Python Types

- Types → A communication method → Communicates behaviors and constranis
	- Mechanical representation: To Python language
	- Semantic representation: To other developers
## Typing systems
### Strong versus weak:
- Python is a strongly typed language.
- However, the `TypeError` arises at runtime instead of development time.

### Dynamic versus static

Static
- Variables do not change its type during runtime.
- Developers might add explicit types to variables.
- Languages with static typing embed their typing information.

Dynamic:
- Python is a dynamically typed languge.
	- Types are known until runtime.
- Embeds type information in the variable with the value itself.
- Variables can change their types at runtime.