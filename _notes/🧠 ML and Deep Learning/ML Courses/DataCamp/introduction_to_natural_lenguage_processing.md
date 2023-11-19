---
---

# Natural language processing in python

#regular_expressions
#NLP
#python 


## Introduction

What is Natural Language Processing?
- Field of study focused on making sense of language
	--> using statistics and computers


### Regular Expressions
#regex -> [[Regular_Expressions_python]]

- Strings with a special syntax
- Allow us to match patterns in other strings

```python
import re

our_string = 'abcdef'
pattern = r'abc'
re.match(pattern,  our_string)
# it returns a match object
```

- 🚨  It's important to prefix the regex with `r` to ensure than the patterns are interpreted in the way we want.

#### Special patterns

| Pattern    | Matches          | Example           |
| ---------- | ---------------- | ----------------- |
| `\w+`      | word             | 'Magic'           |
| `\d`       | digits           | 9                 |
| `\s`       | space            | ' '               |
| `.*`       | wildcard         | 'perroShiquito23' |
| `+` or `+` | greedy match     | 'aaaa'            |
| `\S`       | **not** space    | 'no_spaces'       |
| `[a-z]`    | lowercase groups | 'abcdef'          |


#### Python's `re` module

- `re` module
- `split`, `findall`, `search`, `match`
- Syntax ==> Pattern, Sting
- May return an *iterator*, *string* or a *match* object.

##### Some examples

```python
import re

capitalized_words = r'[A-Z]\w+'
print(re.findall(
	capitalized_words, my_string
))
```

##### Difference between `search()` and `match()`
- Both `search()` and `match()` expect regex patterns.
- 


### Introduction to tokenization
#tokenization

- ==Tokenization== ==>Turning a string of document into **tokens** (smaller chunks)
- One step in preparing a text for NLP
- There are many different rules and theories for *tokenization*
	- -> based in `RegEx` expressions
- Examples:
	- Breaking out words or sentences
	- Separating by punctuation
	- Separating by `#`
- Why tokenize?
	- Matching common words
	- Removing unwanted tokens
	- Easier to map part of documents

### The `nltk` library
- 📕 #nltk library

```python
from nltk.tokenize import word_tokenize

word_tokenize('Hi there!')
```

##### Some NLTK tokenizers
- `sent_tokenize`: tokenize a document into sentences
- `regexp_tokenize`: tokenize a string or document based on a regular expression pattern.
- `TweetTokenizer`: special class just for tweet tokenization.

### Advanced tokenization with `regex`

- OR is representd using `|`
- We can define a group using `()`
- We can define explicit character ranges using `[]`

```python
import re

match_digits_and_words = ('(\d+\\w+)')

re.findall(match_digits_and_words, 'He has 11 cats.')
```


#### Advanced groups

| Pattern        | Matches                              |
| -------------- | ------------------------------------ |
| `[A-Za-Z]`     | Upper and lowercase english alphabet |
| `0-9`          | number from 0 to 9                   |
| `[A-Za-z\-\.]` | Upper and lowercase, `.` and `-`     |
| `(a-z)`        | `a`, `z`, and `-`                    |
| `\s+\|,`       | One or more spaces or a comma        |

***

## Simple Topic-identification

### Bag of words

 - Basic method to find topics in a text
 - The more frequent a word -> the more important the word is ==> Determine significant words in a text

```python
from nltk.tokenize import word_tokenize
from collections import Counter

# Count the number of times a word appears
counter = Counter(
	word_tokenize("""
		The cat is in the box. The cat likes the box. 
		The box is over the cat.
	""")
)

# Get the results
counter.most_common(2)
```


### 🔍 Simple text preprocessing

**Helps make for better input data.**
- **Tokenization**
- **Lowercasing words**
- #Lemmatization or #Stemming ->
	- Shorten words to their root stems
	- `WordNetLemmatizer`
- **Removing:**
	- #Stop_words
	- Punctuation
	- Unwanted tokens 
- **Plural nouns to singular**

##### Stopwords and Lemmatization

- ==Lemmatization== -> the process of converting a word to its base form. It considers the context.
- ==Stemming== -> just remove the last characters but does not consider the context

```python
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

text = """
	The cat is in the box. The cat likes the box. 
	The box is over the cat.
"""

# *** We will remove non alphabetical and 
# *** Stoping words

# keep words in lowercase
tokens = [w for w in word_tokenize(text.lower())
		 			if w.isalpha()	# only alphabetic strings
		        ]

no_stops = [t for t in tokens
		   			if t not in stopwords.words('english')
		   ]

lemmatized = [wordnet_lemmatizer.lemmatize(t)
				for t in no_stops 	
			]

# Create a bag-of-words
bow = Counter(lemmatized)
```

### Introduction to `gensim`
#gensim

- ==Gensim== is apopular open-source #NLP library
- Uses top academic models to perform complex tasks:
	- Building document of **word vectors** => #word-embedding
	- Performing **topic identification** and document comparison
	- #LDA => *Latent dirichlet allocation* -> A statistical model that can be applied to texts
- It allows to build #corpora and dictionaries using simple classes and functions
	- ==corpus== and ==corpora== => A **set of texts to help perform NLP tasks**.

##### Word vectors
- #words-vectors are multi-dimensional mathematical representations of words created using deep learning methods.
- They give us insight into relationships between words in a corpus.

#### Creating a new corpus

- The corpus can be saved and reused
- Can also be updated

```python
from gensim.corpora.dictionary import Dictionary
form nltk.tokenize import word_tokenize

my_documents = ['text1', 'text2', 'el perro está bien loco´
			   'uvachacauva rules', 'Vine a comala porque me..'
				'de los de lanza en astillero, rocin flaco y galgo...'
			   ]

# Perform some preprocessing steps
tokenized_docs = [word_tokenize(doc.lower())
				 for doc in my_documents]

# Create a dictionary
dictionary = Dictionary(tokenized_docs)
dictionary.token2id

# Create a gensim corpus
corpus = [dictionary.doc2bow(doc)
		 for doc in tokenized_docs]
# It returns a list of lists
```

- Inspect the most common terms per document

```python
# Save the fifth document: doc
doc = corpus[4]

# Sort the doc for frequency: bow_doc
bow_doc = sorted(doc, key=lambda w: w[1], reverse=True)

# Print the top 5 words of the document alongside the count
for word_id, word_count in bow_doc[:5]:
	print(dictionary.get(word_id), word_count)
	
# Create the defaultdict: total_word_count
total_word_count = defaultdict(int)
for word_id, word_count in itertools.chain.from_iterable(corpus):
    total_word_count[word_id] += word_count
```

### Tfiidf with genism
#tf-idf -> Term frequency - inverse document frequency

- Allows you to determine the most important words in each document.
- Ensures most common words don't show up as key words.
- Keeps document specific frequent words weighted high

#### Tf-idf formula

Equation to compute $w_{i,j}$ that is the weight of token $i$ in document $j$

$$w_{i,j} = tf_{i,j} * \textrm{log}\left(\frac{N}{df_i}\right)$$

- $w_{i,j}$: tf-idf weight for token $i$ in document $j$
- $tf_{i,j}$: *term frequency* -> number of occurences of toke $i$ in  document $j$
- $df_{i,j}$: number of documents that contain toke $i$
- $N$: total number of documents

##### Interpretation
- ⬇️ Words that **occur across many or all documents** will have a very **low** tf-idf weight.
- ⬆️ Words that appear in a few documents will have higher tf-idf weight.

##### Tf-idf with gensim

```python
from gensim.models.tfidfmosel import TfidfModel

tfidf = TfidfModel(corpus)
# This will relate each dictionary key
# to its repective weight value

# Calculate the tfidf weights of doc: tfidf_weights
tfidf_weights = tfidf[doc]

# Print the first five weights
print(tfidf_weights)

# Sort the weights from highest to lowest: sorted_tfidf_weights
sorted_tfidf_weights = sorted(tfidf_weights, 
    key=lambda w: w[1], reverse=True)

# Print the top 5 weighted words
for term_id, weight in sorted_tfidf_weights[:5]:
    print(dictionary.get(term_id), weight)
```
 
 
 ## Named-entity Recognition
 
 #NER => Named-entity recognition 👩🏻‍⚖️🏛📆🇲🇽🌲->
 - NPL task to identify important named entities in the text
	 - People, places, organizations
	 - Dates, states, works of art
	 - ... many other categories
 - Can be used alongside topic identification.
	 - ...or on its own!
 - Answer:
	 - Who? When? Where?

#### `nltk` and the Stanford CoreNLP Library
- The Stanford CoreNLP library:
	- Integrated into Python via `nltk`
	- Java based
	- Great support for #NER as well as cofeference and dependency trees

**Required to run NLTK along with the Stanford Library**
🚨 NLTK, the Stanford Java Libraries and some environment variables to help with integration.

```python
import nltk
# Tokenize the article into sentences: sentences
sentences = sent_tokenize(article)

# Tokenize each sentence into words: token_sentences
token_sentences = [word_tokenize(sent) for sent in sentences]

# Tag each tokenized sentence into parts of speech: pos_sentences
pos_sentences = [nltk.pos_tag(sent) for sent in token_sentences] 

# Create the named entity chunks: chunked_sentences
chunked_sentences = nltk.ne_chunk_sents(pos_sentences, binary=True)

# Test for stems of the tree with 'NE' tags
for sent in chunked_sentences:
    for chunk in sent:
        if hasattr(chunk, "label") and chunk.label() == "NE":
            print(chunk)

```


#### Charting practice

```python
# Create the defaultdict: ner_categories
ner_categories = defaultdict(int)

# Create the nested for loop
for sent in chunked_sentences:
    for chunk in sent:
        if hasattr(chunk, 'label'):
            ner_categories[chunk.label()] += 1
            
# Create a list from the dictionary keys for the chart labels: labels
labels = list(ner_categories.keys())

# Create a list of the values: values
values = [ner_categories.get(v) for v in labels]

# Create the pie chart
plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)

# Display the chart
plt.show()
```


### Introduction to ==SpaCy==
#spacy => Another great library for #NLP 
- Similar to #gensim but with different implementations
- Focus on creating ==NLP pipelines== to generate models and corpora
- Advantages ✅ :
	- Easy pipeline creation
	- Different entity types compared to `nltk` 
	- **Informal language** corpora
		- -> Easily find entities in Tweets and chat messages
	- Extra categories, not included in `nltk`, such as:
		- `NORP`, `CARDINAL`, `MONEY`, `WORK_OF_ART`, `LANGUAGE`, `EVENT`
- Extra libraries and tools:
	- #Displacy => A [visualization tool](https://explosion.ai/demos/displacy?text=Mi%20perro%20corri%C3%B3%20muy%20r%C3%A1pido%20por%20la%20calle%20Arce%20hacia%20la%20iglesia%20de%20Ensenada&model=es_core_news_sm&cpu=1&cph=1) for viewing parse trees which uses Node.js to create interactive text
		- ==> To label entities and visualize them.

#### Working with `sopacy`

```python
import spacy

# Load a trained word vector
nlp = spacy.load('en'
	tagger=False, parser=False, matcher=False)
# the entity object is an Entity recognizer object
# from the pipeline module
# used to find entities in the text
nlp.entity

# Try it with a document example
my_text = '''The little dog is really crazy coming from
		the New York city'''
doc = nlp(my_text)

# Access to the identified entities
doc.ents

# Access to the labels
print(doc.ents[0], doc.ents[0].label_)

# Print all of the found entities and their labels
for ent in doc.ents:
    print(ent.label_, ent.text)
```


### Multilingual #NER with `Polyglot`
#polyglot => **Another library for NLP** which uses word vectors.
#### Why `polyglot`

- It has vectors for many different languages => more than 130!
- ==Transliteration== is the ability to translate text by swapping character from one language to another.

#### Named Entity Recognition in Spanish
- We need to have the proper vectors downloaded and installed.

```python
from polyglot.text import Text

text = '''El perro chiquito nación en Nueva York
		  la semana pasada. Ha estado haciendo frio.
		  Y el gigante de Galeano ha venido a ver
		  qué andan haciendo todos ustedes aquí en la
		  palza de Mayo, de Buenos Aires.'''

# polyglot automatically detects the language
ptext = Text(text)

# Get the entities
ptext.entities

# Create the list of tuples: entities
entities = [(ent.tag, ' '.join(ent)) for ent in txt.entities]

```

***
## Build a 'Fake news' classifier

### Classifying fake news using supervised learning with NLP

- *How to create supervised learning data from **text**?*
	- `bag-of-words` as features
	- `Tf-idf` as features

### Building a word count vectors with scikit-learn
#### Example #project 🎬
- 🏁  **Goal**: Predicting movie genre from movie plots summary (text).
- 🗂  Data set consisting of movie plots and corresponding gene.
	- *Can we predict genre based on the words used in the plot summary?*
	
##### Count Vectorizer with #scikit-learn
Steps:
1. Load the data and generate the Training and test sets.
2. Use `CountVectorizer` to generate the *Bag-of-words* object => Do that over the *training set* 
		=> Using `.fit_transform()`
1. Apply the fitted `CountVectorizer` over the *test set* 
		=> Using `.transform()` 
		-  🚨 An error could occur if the *test set* has words not included in the training set

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# Load the data
df = ... # load data into DataFrame
y = df['genre_labels'] # indicates if a sci-fi movie

# Preform train_test split
X_train, X_test, y_train, y_test = \
	train_test_split(
		df['plot_text'], y,
		test_size = 0.33
)

# Create the count vectorizer 
# Remove 'stop words' => Similar to Gensim 
count_vectorizer = CountVectoraizer(stop_words='english')
# Each token now acts as a feature

# Create the BOW vectors dictionary
count_train = \ 
	count_vectorizer.fit_transform(X_train.values)

# Use the same dictionary generated with the
# Train set, now with the Test set
count_test = \
count_vectorizer.transform(X_test.values)

```


##### Using `TfidfVectorizer`

```python
# Import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize a TfidfVectorizer object: tfidf_vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Transform the training data: tfidf_train 
tfidf_train = tfidf_vectorizer.fit_transform(X_train)

# Transform the test data: tfidf_test 
tfidf_test = tfidf_vectorizer.transform(X_test)

# Print the first 10 features
print(tfidf_vectorizer.get_feature_names()[:10])

# Print the first 5 vectors of the tfidf training data
print(tfidf_train.A[:5])
```

##### Inspecting the vectors
To get a better idea of how the vectors work, we can investigate the `vectorizers` using pandas.

```python
# Create the CountVectorizer DataFrame: count_df
count_df = pd.DataFrame(count_train.A, 
    columns=count_vectorizer.get_feature_names())

# Create the TfidfVectorizer DataFrame: tfidf_df
tfidf_df = pd.DataFrame(tfidf_train.A,
    columns=tfidf_vectorizer.get_feature_names())

# Print the head of count_df
print(count_df.head())

# Print the head of tfidf_df
print(tfidf_df.head())

# Calculate the difference in columns: difference
difference = set(count_df.columns) - set(tfidf_df.columns)
print(difference)

# Check whether the DataFrames are equal
print(count_df.equals(tfidf_df))

```

### Training and testing a classification model with scikit-learn

Now, we will use the features we have extracted to train and test a supervised classification model.

#### Naive Bayes Classifier for NLP
A #NaiveBayes classifier is commonly used for **testing NLP classification** => Its basis in probability
- *Given a particular piece of data, how likely is a particular outcome?*
- Simple and effective model
- Each word from `CountVectorizer` acts as a feature

##### Back to the example

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# instatntiate the classifier
nb_classifier = MultinomialNB()

# Train it with the Count_vectorizer object
nb_claddifier.fit(cont_train, y_train)

# Make a prediction
pred = nb_classifier.predict(count_test)

# Test the accuracy
metrics.accuracy_score(y_test, pred)
```

#### Inspecting the model
Now we can map the important vector weights back to actual words using the following inspection techniques:

```python
# Get the class labels: class_labels
class_labels = nb_classifier.classes_

# Extract the features: feature_names
feature_names = tfidf_vectorizer.get_feature_names()

# Zip the feature names together with the coefficient array and sort by weights: feat_with_weights
feat_with_weights = sorted(zip(nb_classifier.coef_[0], 
                        feature_names))

# Print the first class label and the top 20 feat_with_weights entries
print(class_labels[0], feat_with_weights[:20])

# Print the second class label and the bottom 20 feat_with_weights entries
print(class_labels[1], feat_with_weights[-20:])

```


### Simple NLP, complex problems

*How complex the problems can be?*

##### Further steps
1. Tweaking hyperparameters
2. Trying a new classification model
3. Training on a large dataset
4. Improving text preprocessing

##### Sentiment Analysis
![[Captura de Pantalla 2021-03-05 a la(s) 12.53.18.png]]

