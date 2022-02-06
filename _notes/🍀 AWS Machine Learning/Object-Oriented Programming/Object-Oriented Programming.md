---
---

# Object-Oriented Programming

## Lesson outline
- Object-oriented programming syntax
	- Procedural vs. object oriented programming
	- Clases, objects, methods and attributes
	- Dosing a class
	- Magic methods
	- Inheritance
- Using object-oriented programming to make a Python package
	- making a package
	- Tour of `scikit-learn` source code
	- Putting your package on PyPi

## Object-Oriented programming
- #OOP allows to create large, modular programs that can easily expand over time
- #OOP hides the implementation from the end user.
- If the implementation changes the user may not notice it

## Procedural vs Object-Oriented programming

### Object oriented programming
- Modeled around ==Objects== =>
	- 🟠  ==Characteristics==: 
		- Attributes `Nouns`: -> Properties of the object
	- 🔵  ==Actions==:  
		- Methods `Verbs`: -> Actions that an object can do

## Class, Object, method, and attribute

#### *Class*
- A blueprint consisting of methods and attributes

#### *Object*
- An *instance* of the class.
- They have they own specific definitions of *attributes*
- Can be very *concrete* or even *abstract*

#### *Attribute*
- A descriptor or a characteristic of the object.
- Can take specific values on each object of the same class
	- Also know as:
		- `property`, `description`, `feature`, `quality`, `trait`, `characteristic`

#### *Methods*
- An action that a class or object could take.

#### Encapsulation
- One of the fundamental ideas behind object-oriented programming is called encapsulation:
	- Combine functions and data all into a single entity.
	- *Encapsulation* allows to hide implementation details

## OOP in python

```python
# Define the class
class Shirt:
	def __init__(self, color, size, style, price):
		# Object attibutes
		self.color = color
		self.size  = size
		self.style = style
		self.price = price
		
	def change_price(self, new_price):
		self.price = new_price
		
	def discount(self, discount):
		self.price = self.price * (1 - discount)
```

#### `self`:
- `self` it is essentially a dictionary that holds the attributes of a `Class` 
	- It makes the attributes available through the class
- `self` tells python where to look in the computer's memory
- It is a *convention* (any other word could be plausible)

#### Object instantiation
```python
new_shirt = Shirt('red', 'S', 'short sleeve', 15)
```

### Set and Get methods
- A `get` method is for obtaining an attribute value
- A `set` method is for changing an attribute value.

#### Private attributes:
- Python does not distinguish between **private** and **public** variables.
- There is a convention for private attributes:
	- Use an underscore before the variable name
		- `name` -> `_name`
	- Nevertheless, the variable is still accessible through: `object._name`