---
---

# Code reviews
Code reviews benefit everyone in a team to promote best programming practices and prepare code for production.
- Benefits:
	- Catch errors
	- Ensure readability
	- Check standards are met
	- Share knowledge among teams

##### Resources
-   [Code reviews](https://github.com/lyst/MakingLyst/tree/master/code-reviews)
-   [Code review best practices](https://www.kevinlondon.com/2015/05/05/code-review-best-practices.html)

### Questions to ask when conducting a code review
#### Is the code clean and modular?
1. Can I understand the code easily?
2. Does it use meningful names and whitespace?
3. Is there duplicated code?
4. Can I provide another layer of abstraction?
5. Is each function and module necessary?
6. Is each function or module too long?

#### Is the code efficient?
1. Are there loops or other steps I can vectorize?
2. Can I use better data structures to optimize any steps?
3. Can I shorten the number of calculations needed for any steps?
4. Can I use generators or multiprocessing to optimize any steps?

#### Is the documentation effective?
1. Are inline comments concise and menaingful?
2. Is there complex code that's missing documentation?
3. Do functions use effective docstrings?
4. Is there necessary project documentation provided?

#### Is the code well tested?
1. Does the code high test coverage?
2. Do tests check for interesting cases?
3. Are the tests readable?
4. Can the tests be made more efficient?

#### Is the logging effective?
1. Are log messages clear, concise, and professional?
2. Do they include all relevant and useful information?
3. Do they use the appropriate logging level?
