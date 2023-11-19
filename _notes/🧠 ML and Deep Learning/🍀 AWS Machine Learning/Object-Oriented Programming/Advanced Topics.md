---
---

# Advanced Topics and Resources

## Packages and Modules

#PythonModule
- A single python file that contains a collection of functions, classes and/or global variables.

#PythonPackage
- A collection of modules placed into a directory.
- It requires a `__init__.py` file inside the directory.

### Steps to make a package
- Make the adequate imports

# R1: Instance, Class, Static Methods

## Overview
```python
class Perro():
	def method(self):
		return 'instance method called', self
	
	@classmethod
	def classmethod(cls):
		return 'class method called', cls
	
	@staticmethod
	def staticmethod():
		return 'static method'
```

## Key Takeaways
- Instance methods need a class instance and can access the instance through `self`
- Class methods don't need a class instance. Thay cant access the instance (`self`) but they have access to the class itself via `clf`
- Static methods don't have access to `cls` nor `self`. They work like regular functions but belong to the class's namespace.
- Static and class methods communicate and (to a certain degree) enforce developer intent about class design. This can have maintenance benefits
- Using `static` methods and `class` methods are ways to communitcate developer intent while enforcing that intent enough to avoid most slip of the mind mistakes and bugs that would break the design

### Instance Methods
-> `Regular intance` method.
- The most comm on method.
- Takes the `self` parameter as argument ->
	- It points to an instance of `MyClass` when the method is called.
	- Can accept extra parameters
- It gives acces to attributes and other methods ==of the same object==
- They can modify:
	- 🔵  Object state
	- 🔴  Class state
- They can access the class itself through the `self.__class_` attribute.

- Python replace the `self` argument with the inctance object, `obj` =>
```python
# This is a calling
obj = MyClass()
obj.method()
```
```
('instance method called', <MyClass instance at 0x10205d190>)
```
```python
# the same as
MyClass.method(obj)
```

- It **cannot be called** from the Class directly without create an object
- But, after the object is created, they can access to the Class
	- Because ==instance methods== can also acces the *class itself* => `self.__class__`


### Class Methods
- They are marked with the `@classmethod` decorator
- They take a `cls` parameter that points to the ==class== (not the object instance) when the method is called.
- It can't modify the object instance state.
- They can only modify class state.
- They allow to define alternative constructors for your classes.
	- Python only allows one `__init__` method per class.
	- Therefore `class` methods allow to include multiple alternative constructors.


```python
# This is a calling
obj = MyClass()
obj.classmethod()
```
```
('instance method called', <classMyClass at 0x10205d190>)
```
- Now it has only access to the class


### Static methods
- Marked with the `@staticmethod` decorator
- Do not takes neither `self` nor `clf`, but it could take an arbitrary number of other parameters.
	- They don't need a reference to an instance.
- it can neither modify object state nor class state.
- Are restricted in what data they can access
- They are primarily a way to ==namespace== ( #PythonNamespace) the methods.

- It can be called from the `Object` without issues:
```python
obj.staticmethod()
'static method called'
```
- However, static methods can neither access the object instance state nor the class state.
- Static methods work as regular functions but belong to the class's (and every instance's) namespace.
	- <mark style='background-color: #FFA793 !important'>Static methods belong to *Class* and *Instance* **namespace**</mark>

### Calling methods
- Only `staticmethod` and `classmethod` can be called directly from the *Class* without creating an *Instance*
	- Because python cannot populate the `self` argument and therefore the call fails.
```python
>>> MyClass.classmethod()
('class method called', <class MyClass at 0x101a2f4c8>)

>>> MyClass.staticmethod()
'static method called'

>>> MyClass.method()
TypeError: unbound method method() must
    be called with MyClass instance as first
    argument (got nothing instead)
```

## Examples
```python
class Mascota:
	def __init__(self, nombre):
		self.name = nombre
	
	def __repr__(self):
		return f'La mascota {self.name!r}'
```

# R2: Class and Instance attributes
- Instance attributes are owned by the specific instances of a class.
	- For two different instances, the attributes are usually different
- Python class and object attributes are stored in separate dictionaries:


##### Attributes at the *Class* level
- They are owned by the class itself
- Shared by all the *instances* of the class
- They are ==defined outside all methods==:
	- Placed at the top
- Can be accessed by `instance` or `class` name.

```python
class Perro:
	# Class attributes
	especie = 'cannis familiaris'
	reino = 'mamalia'
	
	def __init__(self, nombre):
		self.name = nombre
		
## Access
x = A()
y = A()

x.a == y.a
x.a == A.a
```

# R3: A class can inherit from multiple parent classes

#Mixins => ==Multiple inheritance==

> Mixins take various forms depending on the language, but at the end of the day they encapsulate behavior that can be reused in other classes.

- They are not `abstract classes` but they are generally used on their own as normal classes.

### Example: Add a logger to the existing classes

- The first, but bad, option is create the logger inside each class
```python
import logging

class SomeClass(object):
	def __init__(self):
		...
		
		self.logger = logging.getLogger(
			'.'.join([
				self.__module__,
				self.__class__.__name__
			])
	)
	
	def do_the_thing(self):
		try:
			...
		except BadThing:
			self.logger.error('OH NOES')
```

1.  A ==BETTER== option, create a #Mixins =>
```python
import logging

class LoggerMixin(object):
	@property
	def logger(self):
		return logging.getLogger(
			'.'.join([
				self.__module__,
				self.__class__.__name__
			])
		)
```

2. Now add the #Mixins to the existing classes
```python
class EssentialFunctioner(LoggerMixin, object):  
    def do_the_thing(self):  
        try:  
            ...  
        except BadThing:  
            self.logger.error('OH NOES')  
			
class BusinessLogicer(LoggerMixin, object):  
    def __init__(self):  
        super().__init__()  
        self.logger.debug('Giving the logic the business...')
```