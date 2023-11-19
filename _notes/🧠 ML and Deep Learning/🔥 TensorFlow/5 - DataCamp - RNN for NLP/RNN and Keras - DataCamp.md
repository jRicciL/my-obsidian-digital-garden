---
---

# RNN and Keras

#RNN #Keras #Tensorflow 

***

## Introduction
### Applications of machine learning to text data

- **Sentiment Analysis**
- **Multi-class classification**
- **Text generation**
- **Machine neural translation**

### RNN architectures
- Reduces the number of parameters of the model by avoiding one-hot encoding
- Shares weights between different positions of the text

### Types of RNN

1. ==Many to one== => Classification
2. ==Many to many== 
	1. Same length => Used for text generation
	2. Different lengths => Machine Translation
		1. Uses an ==Encoder== and a ==Decoder==

![[Captura de Pantalla 2021-12-26 a la(s) 11.20.23.png|500]]![[Captura de Pantalla 2021-12-22 a la(s) 20.53.53.png]]

##### More details:
- [[1. W1 - Intro Sequence Models]]
- [[2. W1 - Recurrent Neural Network]]
- [[7. W3 - RNN for Time Series]]

#### How to use a pretrained model for sentiment analysis
#SentimentAnalysis

```python
# Inspect the first sentence on `X_test`
print(X_test[0])

# Get the predicion for all the sentences
pred = model.predict(X_test)

# Transform the predition into positive (> 0.5) or negative (<= 0.5)
pred_sentiment = ["positive" if x > 0.5 else "negative" for x in pred]

# Create a data frame with sentences, predictions and true values
result = pd.DataFrame({'sentence': sentences, 'y_pred': pred_sentiment, 'y_true': y_test})

# Print the first lines of the data frame
print(result.head())
```

## Introduction to language models
- Language models ==> ==Probability of a sentence==
	- *What is the probability of each word in a sentence to appear in a given particular order?*
- The **probability of the sentence** is given by a `softmax` function on the output layer of the network.
	
- #Unigram models:
	- Uses the probability of each word inside the document
	- Assumes the probabilities are independent
- #n-grams models:
	- Uses the probability of each word conditional to the previous $N-1$ words.
		- A ==bigram==:
			- $$P(\mathbf{sentence}) = P(w_1)P(w_2 | w_1)P(w_3 | w_3) ...$$
- #SkipGram models:
	- Computes the probability of the ==context== words, or neighboring words, given the ==center word==.

#### Embedding layer
#word-embedding 

![[Captura de Pantalla 2022-02-20 a la(s) 19.11.03.png]]

### Building vocabulary dictionaries

- Get unique words

```python
# Get unique words
unique_words = list(set(text.split(' ')))

# Create the dictionary
# word is the key, index is value
word_to_index = {
	k: v for (v, k) in enumerate(unique_words)
}

# Create a index to words
index_to_word = {
	k: v for (k, v) in enumerate(unique_words)
}
```

### Preprocessing input

#### Creating $X$ and $y$
- ==The objective== is to predict a word ($y$) based on the previous words ($X$)
- Create different subsets of texts by windowing the document `text`

```python
# Initialize variable X and y
X = []
y = []

# Loop over the text: lenght `sentence_size` per time with step equal to `step`
for i in range(0, len(text) - sentence_size, step):
	X.append(text[i:i + sentence_size])
	y.append(text[i + sentence_size])
```

- For characters
```python
# Create lists to keep the sentences and the next character
sentences = []   # ~ Training data
next_chars = []  # ~ Training labels

# Define hyperparameters
step = 2          # ~ Step to take when reading the texts in characters
chars_window = 10 # ~ Number of characters to use to predict the next one  

# Loop over the text: length `chars_window` per time with step equal to `step`
for i in range(0, len(sheldon_quotes) - chars_window, step):
    sentences.append(sheldon_quotes[i:i + chars_window])
    next_chars.append(sheldon_quotes[i + chars_window])

# Print 10 pairs
print_examples(sentences, next_chars, 10)
```

#### Transforming new text

```python
# Loop through the sentences and get indexes
new_text_split = []
for sentence in new_text:
    sent_split = []
    for wd in sentence.split(' '):
        # Set to 0 the index in case the word is not found in the dictionary.
        index = word_to_index.get(wd, 0)
        sent_split.append(index)
    new_text_split.append(sent_split)

# Print the first sentence's indexes
print(new_text_split[0])

# Print the sentence converted using the dictionary
print(' '.join([index_to_word[index] for index in new_text_split[0]]))
```

## Introduction to RNN inside #Keras

#### `keras.models`

![[Captura de Pantalla 2022-02-20 a la(s) 19.31.03.png]]

- The ==Sequential== API =>  `keras.models.Sequential`
	- Each layer is implemented after other

- The ==Model== API => `keras.models.Model`
	- Is a generic definition of a model that is more flexible and allows multiple inputs and outputs

#### `keras.layers`

#### `keras.datasets`
- IMBD Movie reviews
- Reuters newswire

### Creating a model

```python
# Import required modules
from keras.models import Sequential
from keras.layers import Dense

# Instatniate the model class
model = Sequential()

# Add layers
model.add(Dense(64, 
				activation = 'relu', 
				input_dim = 100))
model.add(Dense(1, 
				activation = 'sigmoid'))

# Compile the model
model.compile(
	optimizer = 'adam',
	loss = 'mean_squared_error',
	metrics = ['accuracy']
)


# Train the model
mode.fit(X_train, y_train, 
		 epochs = 10, batch_size = 32)

# Analyse the model performance
model.evaluate(X_test, y_test)
# Returns the loss and accuracy values

# Predict using the model
model.predict(X_test)
```

## Example using the IMDB Sentiment Classification

```python
model = Sequential([
	Embedding((10000, 128)),
	LSTM(128, dropout = 0.2),
	Dense(1, activation = 'sigmoid')
])

# Compile the model
model.compile(loss = 'binary_crossentropy',
			   optimizer = 'adam',
			   metrics = ['accuracy'])
# Train the model
model.fit(x_train, y_train)

# Evaluation
score, accuracy = model.evaluate(x_test, y_test)
```