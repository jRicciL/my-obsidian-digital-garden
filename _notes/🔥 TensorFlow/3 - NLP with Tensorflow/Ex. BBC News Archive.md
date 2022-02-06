---
---

# Ex. BBC news archive
*This week you will build on last week’s exercise where you tokenized words from the BBC news reports dataset. This dataset contains articles that are classified into a number of different categories. See if you can design a neural network that can be trained on this dataset to accurately determine what words determine what category. Create the vecs.tsv and meta.tsv files and load them into the embedding projector.*


#NLP #Tensorflow #Coursera
***

```python
import csv
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# bbc-text.csv
!gdown --id 1rX10xeI3eUJmOLsc4pOPY6AnCLO8DxNj
```


```python
vocab_size = 1_000 # YOUR CODE HERE
embedding_dim = 16 # YOUR CODE HERE
max_length = 120 # YOUR CODE HERE
trunc_type = 'post' # YOUR CODE HERE
padding_type = 'pre' # YOUR CODE HERE
oov_tok = '<OOV>' # YOUR CODE HERE
training_portion = .8
```

```python
sentences = []
labels = []
stopwords = [ "a", "about", "above", "after", "again", "against", "all", "am", 
             "an", "and", "any", "are", "as", "at", "be", "because", "been", 
             "before", "being", "below", "between", "both", "but", "by", "could", 
             "did", "do", "does", "doing", "down", "during", "each", "few", "for", 
             "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", 
             "he's", "her", "here", "here's", "hers", "herself", "him", "himself", 
             "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", 
             "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", 
             "most", "my", "myself", "nor", "of", "on", "once", "only", "or", 
             "other", "ought", "our", "ours", "ourselves", "out", "over", "own", 
             "same", "she", "she'd", "she'll", "she's", "should", "so", "some", 
             "such", "than", "that", "that's", "the", "their", "theirs", "them", 
             "themselves", "then", "there", "there's", "these", "they", "they'd", 
             "they'll", "they're", "they've", "this", "those", "through", "to", 
             "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", 
             "we're", "we've", "were", "what", "what's", "when", "when's", "where", 
             "where's", "which", "while", "who", "who's", "whom", "why", "why's", 
             "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", 
             "yours", "yourself", "yourselves" ]
print(len(stopwords))
# Expected Output
# 153
```

```python
with open("./bbc-text.csv", 'r') as csvfile:
    ### START CODE HERE
    lines = csvfile.readlines()
    for line in lines[1:]:
      c, s = line.split(',')
      for sw in stopwords:
        sw = f' {sw} '
        s = s.replace(sw, ' ').replace(' ', ' ')
      labels.append(c)
      sentences.append(s)
    ### END CODE HERE

    
print(len(labels))
print(len(sentences))
print(sentences[0])
```

```python
train_size = int(len(labels) * training_portion) # YOUR CODE HERE

train_sentences = sentences[:train_size] # YOUR CODE HERE
train_labels = labels[:train_size] # YOUR CODE HERE

validation_sentences = sentences[train_size:]# YOUR CODE HERE
validation_labels = labels[train_size:]# YOUR CODE HERE

print(train_size)
print(len(train_sentences))
print(len(train_labels))
print(len(validation_sentences))
print(len(validation_labels))

# Expected output (if training_portion=.8)
# 1780
# 1780
# 1780
# 445
# 445
```

```python
tokenizer = Tokenizer(num_words=vocab_size,
                      oov_token = oov_tok) # YOUR CODE HERE
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(train_sentences) # YOUR CODE HERE
train_padded = pad_sequences(
    train_sequences,
    maxlen = max_length,
    padding = padding_type
) # YOUR CODE HERE

print(len(train_sequences[0]))
print(len(train_padded[0]))

print(len(train_sequences[1]))
print(len(train_padded[1]))

print(len(train_sequences[10]))
print(len(train_padded[10]))

# Expected Ouput
# 449
# 120
# 200
# 120
# 192
# 120
```

```python
validation_sequences = tokenizer.texts_to_sequences(validation_sentences) # YOUR CODE HERE
validation_padded = pad_sequences(
    validation_sequences,
    maxlen = max_length,
    padding = padding_type
)# YOUR CODE HERE

print(len(validation_sequences))
print(validation_padded.shape)

# Expected output
# 445
# (445, 120)
```

```python
label_tokenizer = Tokenizer()# YOUR CODE HERE
label_tokenizer.fit_on_texts(labels)

training_label_seq = np.array(label_tokenizer\
                              .texts_to_sequences(train_labels))# YOUR CODE HERE
validation_label_seq = np.array(label_tokenizer\
                                .texts_to_sequences(validation_labels)) # YOUR CODE HERE

print(training_label_seq[0])
print(training_label_seq[1])
print(training_label_seq[2])
print((training_label_seq).shape)

print(validation_label_seq[0])
print(validation_label_seq[1])
print(validation_label_seq[2])
print((validation_label_seq).shape)

# Expected output
# [4]
# [2]
# [1]
# (1780, 1)
# [5]
# [4]
# [3]
# (445, 1)
```

```python
model = tf.keras.Sequential([
# YOUR CODE HERE
  tf.keras.layers.Embedding(
      input_dim = vocab_size,
      output_dim = embedding_dim,
      input_length = max_length
  ),
  tf.keras.layers.GlobalAveragePooling1D(),
  tf.keras.layers.Dense(24, activation='relu'),
  tf.keras.layers.Dense(6, activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
```

```python
Expected Output
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        (None, 120, 16)           16000     
_________________________________________________________________
global_average_pooling1d (Gl (None, 16)                0         
_________________________________________________________________
dense (Dense)                (None, 24)                408       
_________________________________________________________________
dense_1 (Dense)              (None, 6)                 150       
=================================================================
Total params: 16,558
Trainable params: 16,558
Non-trainable params: 0
```


```python
num_epochs = 30
history = model.fit(
    train_padded,
    training_label_seq,
    epochs = num_epochs,
    validation_data = (validation_padded, validation_label_seq)
)
```
```python
import matplotlib.pyplot as plt


def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")
```

