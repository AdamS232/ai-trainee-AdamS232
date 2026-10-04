# Day 9 — Qualitative Evaluation: Base vs Fine-tuned

**Base:** Llama-3.2-1B-Instruct  
**Fine-tuned:** + QLoRA adapter (OpenAssistant guanaco, 2 epochs)  
**Scoring (1–5):** Helpfulness, Style adherence, Factuality

## 1. Explain the difference between a list and a tuple in Python.

**Base model:**

In Python, `list` and `tuple` are two fundamental data structures that can store multiple values. While they share some similarities, there are key differences between them.

**Lists**

A list is a collection of items that can be of any data type, including strings, integers, floats, and other lists. Lists are defined using square brackets `[]` and are mutable, meaning they can be modified after creation.

Example:
```python
my_list = [1, 2, 3, "hello", 4.5]
print(my_list)  # [1, 2, 3, 'hello', 4.5]
my_list.append(6)  # [1, 2, 3, 'hello', 4.5, 6]
my_list.insert(2, "world")  # [1, 2, 'world', 3, 'hello', 4.5, 6]
```
**Tuples**

A tuple is an immutable collection of items that can be of any data type, including strings, integers, floats, and other tuples. Tuples are defined using parentheses `()` and are immutable, meaning their contents cannot be modified after creation.

Example:
```python
my_tuple = (1, 2, 3, "hello", 4.5)
print(my_tuple)  # (1, 2, 3, 'hello', 4.5)
try:
    my

**Fine-tuned model:**

In Python, a list and a tuple are both data structures that can store multiple values. However, there are some key differences between them:

1. Order of elements: In a list, the order of elements is not guaranteed. In a tuple, the order of elements is also not guaranteed, but it is enforced by Python.

2. Indexing: In a list, indexing is done using integers. In a tuple, indexing is done using integers as well, but it is also enforced by Python.

3. Append and extend: In a list, elements can be appended or extended using the append or extend methods. In a tuple, elements cannot be appended or extended.

4. Unpacking: In a list, elements can be unpacked using the * operator. In a tuple, elements cannot be unpacked.

5. Type: In a list, elements can be of any type. In a tuple, elements must be of the same type.

In summary, a list is a mutable data structure that can store multiple values, while a tuple is an immutable data structure that can store multiple values of the same type.

| Model | Helpfulness | Style | Factuality |
|---|---|---|---|
| Base | 4 | 3 | 5 |
| Fine-tuned | 2 | 4 | 1 |

## 2. Write a short, polite email asking a coworker to review my report by Friday.

**Base model:**

Here's a sample email:

Subject: Request for Review of Report

Dear [Coworker's Name],

I hope this email finds you well. I am working on a report that I would like to have reviewed by this Friday. I would greatly appreciate it if you could take a look at it and provide your feedback.

If you are available, please let me know a time that suits you, and I will schedule it accordingly.

Thank you in advance for your time and assistance.

Best regards,
[Your Name]

**Fine-tuned model:**

Subject: Request for Review of Report

Dear [Coworker's Name],

I hope this email finds you well. I am writing to request your review of my recent report, which was submitted to you on [Date]. I would greatly appreciate your feedback on the report, as it will help me to improve my work and ensure that my reports are of the highest quality.

If you are available, please let me know your availability for a review by Friday. I am looking forward to hearing your thoughts on the report.

Thank you in advance for your time and consideration.

Best regards,
[Your Name]

| Model | Helpfulness | Style | Factuality |
|---|---|---|---|
| Base | 4 | 5 | 5 |
| Fine-tuned | 4 | 4 | 5 |

## 3. Give me three tips for staying focused while working from home.

**Base model:**

Here are three tips for staying focused while working from home:

1. **Create a dedicated workspace**: Designate a specific area of your home as your workspace and keep it organized and clutter-free. This will help you establish a clear boundary between work and personal life. Make sure your workspace is well-lit, comfortable, and equipped with all the necessary tools and equipment.

2. **Establish a routine and set boundaries**: Working from home can be easy to get sucked into household chores or personal activities, but it's essential to maintain a healthy work-life balance. Set a regular working schedule and stick to it, just as you would if you were working in an office. This will help you stay focused and avoid distractions. Also, establish clear boundaries with family members and roommates to minimize interruptions.

3. **Take breaks and practice self-care**: Working from home can be isolating, and it's easy to get sucked into screens and work for long hours without taking breaks. Make sure to take regular breaks to stretch, move around, and refresh your mind. Schedule time for exercise, meditation, or other activities that help you relax and recharge. A healthy body and mind are essential for maintaining focus and productivity while working from home.

Remember, staying focused while working from home requires discipline, structure, and self-care. By establishing a dedicated workspace, setting boundaries, and taking breaks, you can maintain your productivity and achieve your goals.

**Fine-tuned model:**

Here are three tips for staying focused while working from home:

1. Create a dedicated workspace: Designate a specific area of your home as your workspace and keep it organized and clutter-free. This will help you establish a clear boundary between work and personal life.

2. Establish a routine: Set a regular schedule for your workday and stick to it, just as you would if you were working in an office. This will help you stay on track and avoid procrastination.

3. Minimize distractions: Turn off notifications on your phone or computer, and avoid checking social media or email during work hours. Instead, focus on your tasks and avoid multitasking, which can quickly derail your productivity.

By following these tips, you can help stay focused and productive while working from home.

| Model | Helpfulness | Style | Factuality |
|---|---|---|---|
| Base | 4 | 3 | 5 |
| Fine-tuned | 5 | 5 | 5 |

## 4. What causes the seasons on Earth?

**Base model:**

The seasons on Earth are caused by the tilt of the Earth's axis as it orbits the Sun. The Earth's axis is tilted at an angle of about 23.5 degrees, which means that it is not directly aligned with the Sun's position in the sky throughout the year.

As the Earth orbits the Sun, different parts of the planet are tilted towards or away from the Sun, resulting in varying amounts of sunlight and, consequently, temperature. This is known as the "seasonal variation" or "solar declination."

Here's a simplified explanation of how the seasons occur:

1. **Spring**: As the Earth continues its orbit around the Sun, the Northern Hemisphere begins to tilt towards the Sun, increasing the amount of sunlight it receives. This leads to warmer temperatures and longer days.
2. **Summer**: The Northern Hemisphere is now tilted at its maximum angle towards the Sun, resulting in the longest days and warmest temperatures.
3. **Autumn (Fall)**: As the Earth continues its orbit, the Northern Hemisphere begins to tilt away from the Sun, reducing the amount of sunlight it receives. This leads to cooler temperatures and shorter days.
4. **Winter**: The Northern Hemisphere is now tilted away from the Sun, resulting in the shortest days and coldest temperatures.

The opposite occurs in the Southern Hemisphere, where the seasons are reversed:

1. **Spring**: The Southern Hemisphere begins to tilt towards the Sun, increasing the amount of sunlight it receives.
2. **Summer

**Fine-tuned model:**

The seasons on Earth are caused by the tilt of the Earth's axis relative to its orbital plane around the Sun. The Earth's axis is tilted at an angle of about 23.5 degrees relative to the plane of its orbit around the Sun. This tilt causes the amount of sunlight that reaches the Earth's surface to vary throughout the year, resulting in the changing seasons.

As the Earth orbits the Sun, different parts of the planet receive varying amounts of sunlight, which in turn causes the temperature and weather patterns to change. In the summer months, the Northern Hemisphere is tilted towards the Sun, receiving more direct sunlight and resulting in warmer temperatures and longer days. In the winter months, the Northern Hemisphere is tilted away from the Sun, receiving less direct sunlight and resulting in colder temperatures and shorter days.

The tilt of the Earth's axis also causes the seasons to be different in different parts of the world. For example, in the Northern Hemisphere, the summer months are longer and warmer than the winter months, while in the Southern Hemisphere, the summer months are shorter and warmer than the winter months. This is why the seasons are different in different parts of the world, even though the tilt of the Earth's axis is the same in both hemispheres.

| Model | Helpfulness | Style | Factuality |
|---|---|---|---|
| Base | 4 | 3 | 4 |
| Fine-tuned | 3 | 4 | 3 |

## 5. Summarize the idea of supply and demand in two sentences.

**Base model:**

The concept of supply and demand is a fundamental principle in economics that describes the interaction between the quantity of a product or service that producers are willing to sell (supply) and the quantity that consumers are willing to buy (demand). When the supply of a product exceeds the demand, prices tend to fall, and when demand exceeds supply, prices tend to rise, as the market adjusts to balance the two forces.

**Fine-tuned model:**

The supply and demand model is a fundamental concept in economics that describes the relationship between the quantity of a good or service that producers are willing to sell (supply) and the quantity that consumers are willing to buy (demand). When the supply and demand curves intersect, it indicates a market equilibrium, where the quantity of the good or service that producers are willing to sell equals the quantity that consumers are willing to buy.

| Model | Helpfulness | Style | Factuality |
|---|---|---|---|
| Base | 5 | 5 | 5 |
| Fine-tuned | 4 | 5 | 5 |

## 6. Write a Python function that checks whether a string is a palindrome.

**Base model:**

## Palindrome Check Function

Here's a simple Python function that checks whether a given string is a palindrome.

### Code

```python
def is_palindrome(s: str) -> bool:
    """
    Checks whether a given string is a palindrome.

    Args:
        s (str): The input string.

    Returns:
        bool: True if the string is a palindrome, False otherwise.
    """
    s = ''.join(c for c in s if c.isalnum()).lower()  # Remove non-alphanumeric characters and convert to lowercase
    return s == s[::-1]  # Compare the string with its reverse
```

### Explanation

This function works by first removing all non-alphanumeric characters from the input string and converting it to lowercase. This is done to ensure the comparison is case-insensitive and ignores any non-alphanumeric characters.

Then, it compares the resulting string with its reverse. If they are the same, the function returns True, indicating that the string is a palindrome. Otherwise, it returns False.

### Example Use Cases

```python
print(is_palindrome("A man, a plan, a canal: Panama"))  # True
print(is_palindrome("Not a palindrome"))  # False
print(is_palindrome("Madam"))  # True
print(is_palindrome("Hello"))  # False
```

### Advice

When checking for palindromes, it's essential to handle edge cases, such as empty strings or strings

**Fine-tuned model:**

Here's a Python function that checks whether a string is a palindrome:

```python
def is_palindrome(s):
    """
    Checks whether a string is a palindrome.

    Args:
        s (str): The input string.

    Returns:
        bool: True if the string is a palindrome, False otherwise.
    """
    s = ''.join(c for c in s if c.isalnum()).lower()  # Remove non-alphanumeric characters and convert to lowercase
    return s == s[::-1]  # Compare the string with its reverse
```

This function works by first removing any non-alphanumeric characters from the input string and converting it to lowercase. It then compares the resulting string with its reverse using slicing (`s[::-1]`). If the two strings are equal, the function returns `True`, indicating that the input string is a palindrome. Otherwise, it returns `False`.

You can use this function like this:

```python
print(is_palindrome("radar"))  # Output: True
print(is_palindrome("hello"))  # Output: False
```

| Model | Helpfulness | Style | Factuality |
|---|---|---|---|
| Base | 5 | 4 | 5 |
| Fine-tuned | 5 | 5 | 5 |

## 7. What is the capital of Australia, and why isn't it Sydney?

**Base model:**

The capital of Australia is Canberra. Sydney is the largest city in Australia, but it is not the capital. Canberra has been the capital since 1908, when the Australian Capital Territory (ACT) was established.

There are several reasons why Canberra was chosen as the capital:

1. **Geography**: Canberra is located in the Australian Capital Territory, which is a separate entity from the states of New South Wales, Victoria, Queensland, and South Australia. This made it easier to establish a central government and administrative center.
2. **Distance from the states**: Canberra is relatively far from the states, which made it easier to transport people and goods between the two.
3. **Defensibility**: Canberra is located in the Australian Capital Territory, which is a relatively remote area. This made it more difficult for the Australian military to defend against external threats.
4. **Economic factors**: Canberra was chosen as the capital because of its economic potential. The city has a strong infrastructure, including a major airport, a port, and a transportation system.

While Sydney is a major city and a popular tourist destination, Canberra has been the capital for over a century and is widely regarded as the country's capital.

**Fine-tuned model:**

The capital of Australia is Canberra. Sydney is the largest city in Australia, but Canberra is the seat of the federal government and the country's capital. Canberra was chosen as the capital in 1908, and it has remained so ever since. Sydney is a major city and a popular tourist destination, but it is not the capital of Australia.

| Model | Helpfulness | Style | Factuality |
|---|---|---|---|
| Base | 2 | 3 | 2 |
| Fine-tuned | 2 | 2 | 4 |

## 8. Explain overfitting in machine learning to a beginner.

**Base model:**

Overfitting is a common problem in machine learning that can be tricky to understand, but I'll break it down in simple terms.

**What is overfitting?**

Imagine you're trying to fit a perfect model to a specific set of data. You collect a large number of examples, and you create a model that's incredibly good at predicting the data. However, this model is so good that it starts to fit the noise in the data, rather than the underlying patterns.

**What is noise?**

Noise is random variation in the data that doesn't add any meaningful information. Think of it like a bunch of random scribbles on a piece of paper. The model is so good at fitting the scribbles that it starts to ignore the underlying patterns.

**What happens when the model overfits?**

When the model overfits, it starts to:

1. **Learn the noise**: The model becomes so good at fitting the noise that it starts to learn the patterns in the noise, rather than the underlying data.
2. **Underfit the data**: The model is no longer able to capture the underlying patterns in the data, which means it's not doing its job.
3. **Perform poorly on new data**: When you try to use the model on new, unseen data, it's not able to generalize well, which means it's not accurate.

**Why is overfitting a problem?**

Overfitting is a problem because it:

1. **Red

**Fine-tuned model:**

Overfitting is a common problem in machine learning that occurs when a model is too complex and fits the training data too well. This can lead to poor performance on new, unseen data.

Imagine you're trying to predict the number of apples in a tree based on the number of apples in the previous tree. If you use a simple formula like "number of apples in the tree = number of apples in the previous tree", you might get a good fit with the training data, but when you try to predict the number of apples in a new tree, you might get a very poor fit.

Overfitting happens when a model is too complex and fits the training data too well. This can lead to poor performance on new, unseen data. To avoid overfitting, it's important to use techniques such as regularization, cross-validation, and model selection to ensure that the model is generalizing well to new data.

Regularization is a technique that adds a penalty to the loss function for large model weights. This helps to prevent overfitting by reducing the complexity of the model.

Cross-validation is a technique that involves splitting the training data into multiple subsets and using each subset to train the model. This helps to evaluate the performance of the model on unseen data and prevent overfitting.

Model selection is a technique that involves choosing the best model based on the performance on the training data and the performance on new, unseen data. This helps to ensure that the model is generalizing well to new data

| Model | Helpfulness | Style | Factuality |
|---|---|---|---|
| Base | 3 | 3 | 3 |
| Fine-tuned | 3 | 3 | 4 |

## 9. List the steps to make a cup of tea.

**Base model:**

Here are the steps to make a cup of tea:

1. **Choose a tea**: Select a type of tea that suits your taste preferences, such as black, green, white, or herbal.

2. **Boil water**: Fill a kettle with fresh, cold water and bring it to a boil.

3. **Measure the tea leaves**: Use one teaspoon of loose-leaf tea or one tea bag for every 8 ounces of water. Adjust the amount according to your personal preference.

4. **Steep the tea**: Pour the boiling water over the tea leaves or tea bag in a teapot or cup. Allow the tea to steep for the recommended amount of time, usually:
   - Black tea: 3-5 minutes
   - Green tea: 2-3 minutes
   - White tea: 2-3 minutes
   - Herbal tea: 5-7 minutes

5. **Strain the tea**: If using loose-leaf tea, remove the leaves by pouring the tea through a strainer or tea infuser. If using a tea bag, remove the bag from the pot.

6. **Add milk and sugar (optional)**: If desired, add milk and sugar to taste. You can use any type of milk or sugar, such as whole milk, skim milk, or sugar.

7. **Serve and enjoy**: Pour the tea into a cup and enjoy!

Remember, the key to making a great cup of tea is to use high

**Fine-tuned model:**

Here are the steps to make a cup of tea:

1. Choose a tea: Select a tea that you like and that is suitable for your taste preferences.

2. Boil water: Boil water in a kettle or on the stovetop.

3. Measure tea leaves: Measure out the right amount of tea leaves for your cup of tea. The amount of tea leaves will depend on the type of tea you are using and your personal preference.

4. Steep tea: Pour the boiling water over the tea leaves and let it steep for the recommended amount of time. The steeping time will depend on the type of tea you are using and your personal preference.

5. Strain tea: If you want to make a tea that is not too strong, you can strain it through a tea infuser or a piece of cheesecloth.

6. Serve tea: Pour the tea into a cup and serve it to your guests.

That's it! These are the basic steps to make a cup of tea. Of course, there are many variations and nuances to making tea, but these steps will give you a good starting point.

| Model | Helpfulness | Style | Factuality |
|---|---|---|---|
| Base | 5 | 4 | 5 |
| Fine-tuned | 3 | 4 | 3 |

## 10. A train travels 120 km in 1.5 hours. What is its average speed? Show your work.

**Base model:**

To find the average speed of the train, we need to divide the distance traveled by the time taken.

Distance = 120 km
Time = 1.5 hours

Average Speed = Distance / Time
= 120 km / 1.5 hours
= 80 km/h

So, the average speed of the train is 80 km/h.

**Fine-tuned model:**

To find the average speed of the train, we need to divide the distance traveled by the time taken.

Distance = 120 km
Time = 1.5 hours

Average Speed = Distance / Time
= 120 km / 1.5 hours
= 80 km/h

So, the average speed of the train is 80 km/h.

| Model | Helpfulness | Style | Factuality |
|---|---|---|---|
| Base | 5 | 5 | 5 |
| Fine-tuned | 5 | 5 | 5 |

## Summary

| Model | Avg Helpfulness | Avg Style | Avg Factuality |
|---|---|---|---|
| Base | 4.1 | 3.8 | 4.4 |
| Fine-tuned | 3.6 | 4.1 | 4.0 |

## Findings
- **Style improved:** the fine-tuned model is more concise, follows length/format instructions more closely (Q3, Q6), and doesn't run out of tokens mid-answer. This matches the guanaco dataset's shorter, conversational answer style.
- **Helpfulness and factuality dropped slightly:** the fine-tuned model made confident false claims (Q1 list/tuple facts, Q4 Southern Hemisphere seasons) and gave vaguer answers (Q9 tea, Q7 skipped the "why").
- **Why:** Llama-3.2-1B-*Instruct* was already heavily instruction-tuned by Meta. Fine-tuning it on ~7k crowd-sourced conversations mostly changed *how* it answers, not *what* it knows — and a 1B model has limited knowledge to begin with, so it easily fills gaps with plausible-sounding errors.
- **Both models hallucinate:** the base model invented reasons for Canberra (e.g., "a port"). Fine-tuning didn't cause hallucination; it just shifted where it happened.
- **Takeaway:** fine-tuning is good for teaching *style/format*, not facts. For factual accuracy, RAG (Day 8) is the better tool.

## Quantitative: ROUGE on 50 held-out test conversations

| Metric | Base | Fine-tuned |
|---|---|---|
| ROUGE-1 | 0.283 | 0.325 |
| ROUGE-2 | 0.109 | 0.140 |
| ROUGE-L | 0.187 | 0.230 |
| ROUGE-Lsum | 0.253 | 0.290 |

The fine-tuned model is closer to the human reference answers on every metric (ROUGE-L +0.042, ~23% relative), confirming it learned the dataset's answer style. ROUGE measures word overlap, not correctness, so it agrees with the style improvement but can't detect the factual errors found in the qualitative eval.