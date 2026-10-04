# Day 9 Case Study — Python Tutor Specialist: Base vs Fine-tuned

**System prompt (both models):** You are a senior Python tutor. Explain the concept in exactly two short paragraphs: first what it is and why it matters, then how to use it with a small example.

**Scoring (1–5):** Helpfulness, Style adherence (exactly 2 paragraphs + example), Factuality

**Training:** 100 hand-written examples (`day09/python_tutor.jsonl`, 90 train / 10 eval), QLoRA r=16, 3 epochs, 66 steps, 65 seconds. Train loss 2.86 → 0.78; eval loss 1.207 → **1.182** (epoch 2) → 1.195 (epoch 3, slight overfitting).

**Eval prompts:** 10 topics deliberately *not* in the training data, to test whether the model learned the style rather than memorized answers.

> Note: base answers marked *[cut off]* hit the 300-token generation limit. That's partly a symptom of the base ignoring "two short paragraphs", so it counts against Style.

## 1. What does functools.lru_cache do?

**Base model:**

**What is functools.lru_cache?**

`functools.lru_cache` is a decorator in Python's `functools` module that implements a Least Recently Used (LRU) cache. It's a powerful tool for memoizing functions, which means it stores the results of expensive function calls and returns the cached result when the same inputs occur again. This can significantly speed up the execution time of functions that are called repeatedly with the same arguments.

**Why is it important?**

Using `functools.lru_cache` is crucial in many applications, especially when dealing with computationally expensive functions or data-intensive operations. By caching the results of these functions, you can avoid redundant computations and improve the overall performance of your code. This is particularly useful in scientific computing, data analysis, machine learning, and other fields where computational efficiency is critical.

**Example: Using functools.lru_cache**

Here's a simple example of how to use `functools.lru_cache` to memoize a function that calculates the factorial of a number:
```python
import functools

def factorial(n):
    if n < 2:
        return 1
    return n * functools.lru_cache()(factorial(n-1))

print(factorial(5))  # Output: 120
print(factorial(3))  # Output: 6
```
In this example, the `factorial` function is memoized using `functools.lru_cache`, which stores *[cut off]*

**Fine-tuned (Python tutor):**

`functools.lru_cache` memoizes function results so that repeated calls return the same cached value. It's particularly useful for expensive function calls, like calculating the factorial of a large number.

For example, `@lru_cache(maxsize=32) def factorial(n): return n * factorial(n - 1))` and `print(factorial(5))` and `print(factorial(5))` both give the same result. You can also pass `cache=True` to `lru_cache` to include the cache in the output.

| Model | Helpfulness | Style | Factuality |
|---|---|---|---|
| Base | 3 | 2 | 3 |
| Fine-tuned | 2 | 5 | 2 |

**Notes:** Both explain memoization correctly. Base's code is wrong (it calls `lru_cache()` on a *result*, not as a decorator) and the answer is far too long. Fine-tuned has the right idea (`@lru_cache(maxsize=32)`), but its function has no base case and an extra `)`, and `cache=True` is an invented parameter.

## 2. What is the difference between append and extend on a list?

**Base model:**

**What is the difference between append and extend on a list?**

In Python, `append` and `extend` are two methods used to add elements to the end of a list. The main difference between them is when they are used:

- `append` is used to add a single element to the end of a list. It takes one argument, which is the element to be added.
- `extend` is used to add multiple elements to the end of a list. It takes an iterable (such as a list, tuple, or string) as an argument, which contains the elements to be added.

**Using append and extend with a small example**

Let's say we have a list `fruits` and we want to add some new fruits to it. We can use `append` to add a single fruit, or `extend` to add multiple fruits.

```python
# Using append
fruits = ['apple', 'banana', 'cherry']
fruits.append('orange')
print(fruits)  # Output: ['apple', 'banana', 'cherry', 'orange']

# Using extend
fruits = ['apple', 'banana', 'cherry']
fruits.extend(['orange', 'grape','mango'])
print(fruits)  # Output: ['apple', 'banana', 'cherry', 'orange', 'grape','mango']
```

In this example, `append` adds a single fruit 'orange' *[cut off]*

**Fine-tuned (Python tutor):**

`append` adds a single item to the end, while `extend` adds multiple items in one step. Use `append` when you need to add a single item, and `extend` when you need to add many.

For example, `my_list = [1, 2]` and `my_list.append(3)` adds `3` to the end. But `my_list.extend([4, 5])` adds `4` and `5` in one step, which is usually faster.

| Model | Helpfulness | Style | Factuality |
|---|---|---|---|
| Base | 5 | 2 | 5 |
| Fine-tuned | 4 | 5 | 4 |

**Notes:** Both are correct. Base is thorough with working code but long and cut off. Fine-tuned is concise and in the exact format, but misses the classic gotcha: `append([4, 5])` would add the whole list as a single item.

## 3. What are abstract base classes (abc) in Python?

**Base model:**

**What are Abstract Base Classes (ABCs) in Python?**

Abstract Base Classes (ABCs) in Python are a way to define a blueprint or a base class that cannot be instantiated directly. They are used to create classes that are intended to be inherited by other classes, but are not meant to be used as-is. ABCs are useful when you want to define a common interface or set of methods that must be implemented by derived classes, but do not have their own implementation.

**Why do they matter?**

Using ABCs is important because it allows you to define a common base class that can be used by multiple derived classes, without having to duplicate code. This makes your code more modular, maintainable, and easier to extend. Additionally, ABCs provide a way to define a contract or a set of methods that must be implemented by derived classes, which can help catch errors at compile-time rather than runtime.

**How to use ABCs with a small example**

Here's an example of how to use ABCs to define a base class `Shape` that has methods for calculating the area and perimeter of a shape:
```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass

    @abstractmethod
    def perimeter(self):
        pass
```
We can then define two derived classes, `Circle` and `Rectangle`, that implement the `Shape` interface:
```python
class Circle(Shape
```
*[cut off]*

**Fine-tuned (Python tutor):**

An abstract base class is a class that cannot be instantiated on its own and must be inherited by another class. It defines methods that must be implemented, and any subclass must provide an implementation for those methods.

For example, `Shape` could be an abstract base class with `area` and `perimeter` methods, and `Circle` could inherit from it with its own `radius` method. You can also add a `__new__` method to create a new class from an abstract base class, which is why `@abc.register` works. Use `@abc.abstractmethod` to mark a method as abstract, and `@abc.init` to specify what arguments a class must take.

| Model | Helpfulness | Style | Factuality |
|---|---|---|---|
| Base | 4 | 2 | 4 |
| Fine-tuned | 2 | 5 | 1 |

**Notes:** Base gives a correct, working `ABC`/`@abstractmethod` example (its "compile-time" claim is wrong; Python catches it when you try to instantiate). Fine-tuned's first paragraph is right, but the second invents things: `@abc.init` doesn't exist, and the `__new__`/`@abc.register` explanation is nonsense.

## 4. What is __slots__ and why would you use it?

**Base model:**

**What is __slots__?**

`__slots__` is a special attribute in Python classes that allows you to specify the names of the instance variables (also known as attributes or properties) that a class should use. It's a way to explicitly define the names of the variables that a class should use, rather than relying on Python's default behavior of using arbitrary names. This can be useful when you need to ensure that certain variables are used consistently throughout your code.

**Why use __slots__?**

Using `__slots__` can help you in several ways:

*   **Improved code readability**: By specifying the names of the variables, you can make your code more readable and self-explanatory.
*   **Reduced namespace pollution**: When you use `__slots__`, you avoid the possibility of naming conflicts between variables with the same name.
*   **Better performance**: In some cases, using `__slots__` can improve performance by reducing the number of attribute lookups.

Here's an example of how you can use `__slots__` in a class:

```python
class Person:
    __slots__ = ('name', 'age')

    def __init__(self, name, age):
        self.name = name
        self.age = age

# Create an instance of the Person class
person = Person('John Doe', 30)

# Accessing the attributes
print(person.name)  # Output: John Doe
print(person.age)
```
*[cut off]*

**Fine-tuned (Python tutor):**

`__slots__` is a special attribute that lets you define a class without a `__init__` method, which is a good practice because it makes the code more readable and easier to debug. It's a dictionary of names that you can define in the class definition, and you can use `__slots__ = ("name", "age")` to limit the attributes.

For example, `class Person: __slots__ = ("name", "age")` lets you define a `Person` class without an `__init__` method, and then you can access `person.name` and `person.age` directly. This makes your code more efficient because it avoids the overhead of a `__init__` method. Use it sparingly, because it can make your code harder to understand if you're using it too much.

| Model | Helpfulness | Style | Factuality |
|---|---|---|---|
| Base | 2 | 2 | 2 |
| Fine-tuned | 1 | 5 | 1 |

**Notes:** Neither gives the real reason for `__slots__`: it saves memory by removing each object's `__dict__`, and blocks adding new attributes. Base's example code is correct but its reasons (readability, namespace pollution) are wrong. Fine-tuned is mostly false: `__slots__` has nothing to do with `__init__`, and it isn't a dictionary.

## 5. How does argparse help with command-line scripts?

**Base model:**

**What is Argparse?**

Argparse is a built-in Python module that allows you to create command-line interfaces (CLI) for your scripts. It provides a simple and flexible way to parse command-line arguments, validate input, and display help messages. Argparse is particularly useful when you need to create scripts that require user input, such as configuration files, data processing, or data analysis.

**Why is Argparse important?**

Argparse is essential for creating robust and user-friendly CLI scripts. Without it, you would need to manually parse command-line arguments, which can be error-prone and difficult to maintain. Argparse simplifies the process by providing a standardized way to handle command-line arguments, making it easier to write scripts that are easy to understand and use. Additionally, Argparse helps ensure that your script is well-documented and follows best practices for writing CLI scripts.

**Using Argparse with a small example**

Here's a simple example of how to use Argparse to create a CLI script that asks the user for their name and age, and then displays a personalized greeting:
```python
import argparse

def main():
    parser = argparse.ArgumentParser(description='Print a personalized greeting')
    parser.add_argument('name', help='Your name')
    parser.add_argument('-a', '--age', type=int, help='Your age')
    args = parser.parse_args()

    if args.age is not None:
        print(f'Hello, {args.name}! You are {args
```
*[cut off]*

**Fine-tuned (Python tutor):**

`argparse` parses command-line arguments and provides a flexible way to define options, such as `--help` or `--version`. It also lets you specify default values and error messages.

For example, `def main(args): print("Hello", args["name"])` and `args = parser.parse_args(["-n", "John"])` work perfectly with `argparse`-defined options. You can add help text with `--help` and let `argparse` display it automatically.

| Model | Helpfulness | Style | Factuality |
|---|---|---|---|
| Base | 4 | 2 | 5 |
| Fine-tuned | 2 | 5 | 2 |

**Notes:** Base's explanation and code (`ArgumentParser`, `add_argument`, `parse_args`) are correct. Fine-tuned's summary is fine, but its example skips creating the parser and adding the `-n` option, and uses `args["name"]` (should be `args.name`), so it wouldn't work.

## 6. What is the difference between np.concatenate and np.stack?

**Base model:**

**What is the difference between np.concatenate and np.stack?**

`np.concatenate` and `np.stack` are two popular functions in NumPy that are used to concatenate arrays. While they share some similarities, there are key differences between them.

`np.concatenate` is used to concatenate arrays along a specific axis, whereas `np.stack` is used to stack arrays along a specific axis. This means that `np.concatenate` will stack arrays vertically, while `np.stack` will stack arrays horizontally.

**Why is it important to know the difference?**

Understanding the difference between `np.concatenate` and `np.stack` is crucial because it allows you to choose the most suitable function for your specific use case. For example, if you have two arrays with different shapes, you may want to use `np.stack` to stack them horizontally, while if you have two arrays with the same shape, you may want to use `np.concatenate` to concatenate them vertically.

**Example:**

Here's an example that demonstrates the difference between `np.concatenate` and `np.stack`:

```python
import numpy as np

# Create two arrays
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

# Concatenate along axis 0 (vertical)
concatenate_vertical = np.concatenate((arr1, arr2))
print(concatenate_vertical)  # Output: [1 2 3 4 5
```
*[cut off]*

**Fine-tuned (Python tutor):**

`np.concatenate` stacks two arrays side by side, while `np.stack` stacks them vertically. The choice depends on how many dimensions your data has.

For example, `np.stack([1, 2], 0)` stacks two 1D arrays horizontally, while `np.concatenate([np.array([1, 2]), np.array([3, 4])], 0)` stacks two 2D arrays vertically. Use `np.stack` when you need to stack multiple arrays of the same size, and `np.concatenate` when you need to stack arrays of different sizes.

| Model | Helpfulness | Style | Factuality |
|---|---|---|---|
| Base | 2 | 2 | 2 |
| Fine-tuned | 1 | 5 | 1 |

**Notes:** Both miss the real difference: `concatenate` joins along an *existing* axis (two `(3,)` arrays → `(6,)`), while `stack` creates a *new* axis (→ `(2, 3)`), and `stack` requires identical shapes. Base's "vertical vs horizontal" framing is wrong, though its one code line is correct. Fine-tuned's examples are wrong (`np.stack([1, 2], 0)` stacks two scalars; the `concatenate` example uses 1D arrays, not 2D).

## 7. What does np.argmax return?

**Base model:**

**What is np.argmax?**

`np.argmax` is a function in NumPy (Numerical Python) that returns the index of the maximum value in an array. It is a vectorized operation, meaning it operates on entire arrays at once, without the need for loops or conditional statements. This makes it much faster and more efficient than using a for loop to find the maximum value.

**Why is it important?**

`np.argmax` is important because it allows you to easily find the index of the maximum value in an array, which can be useful in a variety of applications, such as data analysis, machine learning, and scientific computing. For example, if you have an array of exam scores and want to know which student scored the highest, you can use `np.argmax` to find the index of the maximum score.

**Example:**

Here's an example of how to use `np.argmax` to find the index of the maximum value in an array:
```python
import numpy as np

# Create an array of exam scores
scores = np.array([90, 85, 95, 80, 92, 88, 76, 94, 89, 91])

# Find the index of the maximum value
max_index = np.argmax(scores)

print("The student with the highest score is at index", max_index)
```
This code creates an array of exam scores and uses `np.argmax` to find the index of the maximum value *[cut off]*

**Fine-tuned (Python tutor):**

`np.argmax` returns the index of the maximum value along a given axis. It's like finding the row with the highest score in a vector.

For example, `np.argmax(scores, axis=0)` gives you the row with the highest score in each row, and `np.argmax(scores, axis=1)` gives you the row with the highest score in each column.

| Model | Helpfulness | Style | Factuality |
|---|---|---|---|
| Base | 4 | 2 | 5 |
| Fine-tuned | 2 | 5 | 2 |

**Notes:** Base is correct with a clear, working example (index 2, the 95). Fine-tuned's first sentence is right, but its axis explanation is backwards: `axis=0` gives the row index of the max *in each column*. It also never mentions that with no axis it returns an index into the flattened array.

## 8. What is the difference between pd.concat and pd.merge?

**Base model:**

**What is pd.concat?**

`pd.concat` is a function in pandas that combines two or more dataframes into a single dataframe. It allows you to concatenate (join) two or more dataframes based on a common column. This can be useful when you want to combine data from multiple sources into a single dataset.

**What is pd.merge?**

`pd.merge` is a function in pandas that combines two dataframes based on a common column. It allows you to merge two dataframes on one or more common columns, creating a new dataframe with the combined data. This can be useful when you want to perform an inner join, left join, right join, or outer join on a common column.

**Key differences:**

* `pd.concat` combines dataframes without merging them, while `pd.merge` merges dataframes based on a common column.
* `pd.concat` is faster and more memory-efficient than `pd.merge`, especially when combining large dataframes.
* `pd.merge` allows for more flexibility in the type of join (inner, left, right, outer) and the type of merge (left, right, outer).

**Example:**

Suppose we have two dataframes, `df1` and `df2`, with a common column `id`. We want to combine them into a single dataframe `df3` where each row in `df3` contains all the columns from `df1` and `df2`.

```python
import pandas
```
*[cut off]*

**Fine-tuned (Python tutor):**

`pd.concat` stacks two DataFrames side by side, while `pd.merge` joins them on a common column. Use `concat` when you want to combine two DataFrames in one place, and `merge` when you need to join them separately.

For example, `df1 = pd.DataFrame({"a": [1, 2]})` and `df2 = pd.DataFrame({"b": [3, 4]})` with `df1.merge(df2, on="a")` gives you one row per pair. Use `pd.concat([df1, df2], axis=0)` to stack two DataFrames vertically.

| Model | Helpfulness | Style | Factuality |
|---|---|---|---|
| Base | 3 | 2 | 2 |
| Fine-tuned | 2 | 5 | 2 |

**Notes:** The key idea is that `concat` glues DataFrames along an axis (stacking rows or columns), while `merge` joins rows by matching key values like a SQL join. Base contradicts itself (says `concat` joins "based on a common column") and never shows code. Fine-tuned gets the one-line contrast mostly right, but its `merge(on="a")` example would fail because `df2` has no `a` column.

## 9. What is a set comprehension?

**Base model:**

**What is a Set Comprehension?**

A set comprehension is a concise way to create a new set from an existing iterable (such as a list, tuple, or string) by applying a transformation to each element. It's a powerful feature in Python that allows you to perform complex set operations in a single line of code. Set comprehensions are often used to simplify code, improve readability, and reduce the amount of boilerplate code.

**Why is it Mattering?**

Set comprehensions are useful in a variety of situations, such as:

* Creating sets of unique elements from a list
* Filtering out unwanted elements from a list
* Performing set operations, like union, intersection, and difference
* Creating sets of elements that satisfy a condition

**Example: Creating a Set of Unique Numbers from a List**

Suppose we have a list of numbers: `[1, 2, 2, 3, 4, 4, 5]`. We can use a set comprehension to create a set of unique numbers:
```python
numbers = [1, 2, 2, 3, 4, 4, 5]
unique_numbers = {num for num in numbers if num not in set(numbers)}
print(unique_numbers)  # Output: {1, 2, 3, 4, 5}
```
In this example, the set comprehension iterates over the `numbers` list, checks if each element is already *[cut off]*

**Fine-tuned (Python tutor):**

A set comprehension builds a new set from an existing iterable by applying a transformation to each item. It's a concise way to create a new collection without a loop.

For example, `s = {x**2 for x in range(10)}` gives you a set of squares. Set comprehensions are often faster than lists because they avoid the overhead of a loop. They also work with any iterable, not just lists.

| Model | Helpfulness | Style | Factuality |
|---|---|---|---|
| Base | 2 | 2 | 1 |
| Fine-tuned | 4 | 5 | 3 |

**Notes:** Fine-tuned gives a correct definition and a correct, simple example. Its "faster than lists" claim is wrong, and it never says the key point: duplicates are removed. Base's code is broken: every `num` *is* in `set(numbers)`, so the result is an empty set, not `{1, 2, 3, 4, 5}`.

## 10. Why does collections.OrderedDict still exist now that dicts keep insertion order?

**Base model:**

**What is collections.OrderedDict?**

`collections.OrderedDict` is a dictionary subclass that remembers the order in which keys were inserted. It's a convenient way to maintain a sorted dictionary, where the order of keys is preserved even after the dictionary is modified. This is useful when you need to perform operations on the dictionary, such as iterating over its items in a specific order.

**Why still exist?**

Although dictionaries in Python 3.7 and later versions maintain insertion order by default, `collections.OrderedDict` still exists for backwards compatibility and to provide a way to work with older versions of Python. This is because the `collections` module is a third-party library, and its API is not guaranteed to be stable across all Python versions. By using `collections.OrderedDict`, you can ensure that your code works correctly even if the `collections` module changes in the future.

**Example usage:**

Here's an example of how to use `collections.OrderedDict` to maintain a sorted dictionary:
```python
from collections import OrderedDict

# Create an ordered dictionary
odict = OrderedDict()
odict['apple'] = 1
odict['banana'] = 2
odict['cherry'] = 3

# Print the dictionary in insertion order
print(odict)  # Output: OrderedDict([('cherry', 3), ('apple', 1), ('banana', 2)])

# Modify the dictionary
odict['date'] =
```
*[cut off]*

**Fine-tuned (Python tutor):**

A `OrderedDict` is a dictionary that remembers the order of items inserted, even though Python's built-in `dict` does not. It's still supported because many applications still use `OrderedDict` for its predictable ordering.

For example, with `from collections import OrderedDict; d = OrderedDict([3, 2, 1]))`, you can iterate over the items in the same way as with a regular `dict`. You can also use `OrderedDict.fromkeys(keys)` to create an ordered dictionary from a list of keys.

| Model | Helpfulness | Style | Factuality |
|---|---|---|---|
| Base | 2 | 2 | 1 |
| Fine-tuned | 1 | 5 | 1 |

**Notes:** The real reasons: `OrderedDict` has `move_to_end()` and `popitem(last=False)`, and its `==` comparison is order-sensitive. Neither model says this. Base mentions backwards compatibility (partly fair), but claims `collections` is "third-party" (it's the standard library) and shows a wrong output order. Fine-tuned contradicts the question by saying `dict` doesn't keep order, and its code (`OrderedDict([3, 2, 1]))`) is invalid.

## Summary

| Model | Avg Helpfulness | Avg Style | Avg Factuality |
|---|---|---|---|
| Base | 3.1 | 2.0 | 3.0 |
| Fine-tuned | 2.1 | 5.0 | 1.9 |

**Head-to-head (total of the 3 scores):** fine-tuned 7 wins (Q1, Q2, Q4, Q6, Q8, Q9, Q10), base 3 wins (Q3, Q5, Q7). But on **helpfulness + factuality alone**, base wins 9 of 10; fine-tuned only wins Q9.

## Findings

- **Style transfer worked perfectly:** with only 100 examples and 65 seconds of training, the fine-tuned model followed the "exactly two short paragraphs" format on **10/10** unseen questions. The base model ignored it on 10/10 (headers, bullet lists, long code blocks, cut off at 300 tokens), even with the same system prompt.
- **Knowledge did not transfer, and factuality got worse:** compressing every answer into two paragraphs with inline code pushed the 1B model to write confident but invented details (`@abc.init`, `cache=True`, a broken `merge` example). The base model's long code blocks were more often correct.
- **Why:** these 10 topics were deliberately *not* in the training data. A 100-example dataset can teach format and tone, but it can't teach facts the model doesn't already know, and a 1B model knows little about niche features like `__slots__` or `OrderedDict.move_to_end`.
- **The checklist item ("better on ≥ 6/10"):** met 7/10 on total score, but that win is almost entirely from Style. On substance, base is better. Reported transparently rather than claimed as a clean win.
- **Overfitting:** eval loss was best after epoch 2 (1.182) and rose in epoch 3 (1.195); 2 epochs would have been the better choice.
- **Takeaway:** fine-tune to control *how* a model answers (format, tone, persona); use RAG (Day 8) or a larger model to control *what* it knows. In a real product, combining both (a fine-tuned style plus retrieved documentation) would fix most of these factual errors.