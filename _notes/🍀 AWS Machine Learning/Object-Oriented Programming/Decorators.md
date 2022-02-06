---
---

# Python decorators
#HigherOrderFunctions
#PythonDecorators
***


- Decorators provide a simple syntax for calling ==higher-order functions==:	
	- Functions that take one or more functions as arguments
	- Functions that return a function as its results

- A decorator is a function that takes another function and extends the behavior of the latter function without explicitly modifying it.'
- In python, functions are *first-class objects*

### First-class objects
- Functions are first-class objects:
	- -> *Functions can be passed around and used as arguments*

### Inner Functions
- It is possible to define functions inside other functions.
	- They are not defined until the parent function is called.
	- They scope is limited to the parent => Local variables


### Simple decorators
- <mark style='background-color: #93EBFF !important'>Decorators wrap a function, modifying its behavior</mark>
- In python, decorators are represented by `@`, sometimes called ==pie syntax==.

Without decorator:
```python
def my_decorator(func):
    def wrapper():
        print("Something is happening before the function is called.")
        func()
        print("Something is happening after the function is called.")
    return wrapper

def say_whee():
    print("Whee!")

say_whee = my_decorator(say_whee)

say_whee
```

With a decorator:
```python
def my_decorator(func):
    def wrapper():
        print("Something is happening before the function is called.")
        func()
        print("Something is happening after the function is called.")
    return wrapper

@my_decorator
def say_whee():
    print("Whee!")
	
say_whee
```

### Reusing Decorators
- A decorator is just a regular python function.
- The inner function is usually named as `wrapper()` and the decorated function is names as `func()`

### Decorating functions with arguments
- Use `*args` and `**kwargs` to use an arbitrary number of positional and keyword arguments.

```python
def do_twice(func):
    def wrapper_do_twice(*args, **kwargs):
        func(*args, **kwargs)
        func(*args, **kwargs)
    return wrapper_do_twice
```

### Returning values form decorated functions
- The wrapper function should `return` the value, not the inner function.

```python
def do_twice(func):
    def wrapper_do_twice(*args, **kwargs):
        func(*args, **kwargs)
        return func(*args, **kwargs)
    return wrapper_do_twice
```

### `Intorspection` in python
- It is the ability of an object to know about its own attributes at runtime.
- The `@functools.wraps` decorator preserve information about the original function.
	- It should be passed to the decorated function

```python
import functools

def do_twice(func):
	@functools.wraps(func)
	def wrapper_do_twice(*args, **kwargs):
		func(*args, **kwargs)
		return func(*args, **kwargs)
	return wrapper_do_twice
```

## Decorator Examples


<div class="rich-link-card-container"><a class="rich-link-card" href="https://realpython.com/primer-on-python-decorators/" target="_blank">
	<div class="rich-link-image-container">
		<div class="rich-link-image" style="background-image: url('https://files.realpython.com/media/Primer-on-Python-Decorators_Watermarked.d0da542fa3fc.jpg')">
	</div>
	</div>
	<div class="rich-link-card-text">
		<h1 class="rich-link-card-title">Primer on Python Decorators – Real Python</h1>
		<p class="rich-link-card-description">
		In this introductory tutorial, we'll look at what Python decorators are and how to create and use them.
		</p>
		<p class="rich-link-href">
		https://realpython.com/primer-on-python-decorators/
		</p>
	</div>
</a></div>



### Timer functions
- A function that measures the execution time of a function

```python
import functools
import time

def timer(func):
	
	@functools.wraps(func)
	def wrapper_timer(*args, **kwargs):
		start_time = time.perf_counter()
		func_return = func(*arg, **kwargs)
		end_time = time.perf_counter()
		run_time = end_time - start_time
		# side effect
		print(f"Finished {func.__name__!r} in {run_time}")
		# Return of the decorated function
		return func_return
	return wrapper_timer

@timer
def waste_some_time(num_times):
    for _ in range(num_times):
        sum([i**2 for i in range(10000)])
		
```