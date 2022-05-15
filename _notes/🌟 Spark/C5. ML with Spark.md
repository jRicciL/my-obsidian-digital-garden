---
---

# Machine Learning with Spark
![[Captura de Pantalla 2021-03-21 a la(s) 23.04.44.png]]

## Machine Learning and Spark

### Introduction

Spark is a general purpose framework for cluster computing.
- Compute across a distributed cluster
- Data processed in memory
- Well documented high-level API

#### Spark Cluster components

![[Captura de Pantalla 2021-03-21 a la(s) 23.06.32.png]]

### Connecting to Spark
- Connections with Spark is established by the driver => Can be written in:
	- Java
	- R
	- Scale
	- Python

#### Spark submodules
- Structured Data -> `pyspark.sql`
- Streaming Data -> `pyspark.streaming`
- Machine Learning -> `pyspark.ml`

#### Connection to the cluster
- Remote Cluster using spark URL: `spark://<IP address | DNS name>:<port>`
	- `spark://ec2-18-188-22-23.us-east-2.compute.amazonaws.com:7077`
- Local cluster: `local[n]` where n is the number of cores (optional)

##### Create a #SparkSession

Create a local cluster using `SoarkSession` builder:
```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
			.master('local[*]') \
			.appName('spark_app') \	
			.getOrCreate()
```

Close the session
```python
spark.stop()
```

### Loading data
- Spark represents data using `DataFrame` class.
- Selected methods:
	- `count()`, `show()`, `printSchema()`
- Selected attributes:
	- `dtypes`

```python
cars = spark.read.csv('path_to_csv.csv',
	 header = True,
	 inferSchema = True, # infer data types
	 nullValue = 'NA' # Placeholder for null
	 )

# Get number of records
print("The data contain %d records." % flights.count())

# View the first five records
flights.show(5)

# Check column data types
print(flights.dtypes)
```

- The csv method treads all columns as strings by default.

##### Parse the types of the columns

```python
from pyspark.sql.types import StructType, \
    StructField, IntegerType, StringType

# Specify column names and types
schema = StructType([
    StructField("id", IntegerType()),
    StructField("text", StringType()),
    StructField("label", IntegerType())
])

# Load data from a delimited file
# No header record and fields separated by a semicolon
sms = spark.read.csv('sms.csv', sep=';', 
                header=False, schema=schema)

# Print schema of DataFrame
sms.printSchema()
```

***
## Classification

### Data Preparation

#### Select and Dropping columns

```python
# Load the data
cars = spark.read.csv('path_to_csv.csv',
	 header = True,
	 inferSchema = True, # infer data types
	 nullValue = 'NA' # Placeholder for null
	 )

# Select the deseable columns
cars = cars.select('origin', 'marker', 'size')

# Drop unwnated columns
cars = cars.drop('marker')
```

#### Filtering out missing data
```python
# Count how many missing values are there
cars.filter('cyl IS NULL').count()

# Remove the record with the missing value
# 1)
cars = cars.filter('cyl IS NOT NULL')

# 2) carefull
cars = cars.dropna()
```

#### Mutating columns
```python
from pyspark.sql.functions import round

# Create a new column => 'mass'
cars = cars.withColumn(
	'mass',  # Nomber de la columna
	round(cars.weight / 2.205, 0) # Columna nueva dependiente de otra
			  )
```

```python
# Import the required function
from pyspark.sql.functions import round

# Convert 'mile' to 'km' and drop 'mile' column
flights_km = flights.withColumn('km', 
				round(flights.mile * 1.60934, 0)) \
                .drop('mile')

# Create 'label' column indicating whether flight delayed (1) or not (0)
flights_km = flights_km.withColumn('label', (flights_km.delay >= 15).cast('integer'))

# Check first five records
flights_km.show(5)
```

#### Indexing categorical data
```python
from pyspark.ml.feature import StringIndexer

indexer = stringIndexer(
	inputCol = 'type', # name of the column to index
	outputCol = 'type_idx' # name of the new column
	)
# Perform the indexing
indexer = indexer.fit(cars)

# Create column with index values
cars = indexer.transform(cars)
# By default the strings will be ordered by frequency
```
![[Captura de Pantalla 2021-03-23 a la(s) 18.33.19.png]]

```python
from pyspark.ml.feature import StringIndexer

# Create an indexer
indexer = StringIndexer(inputCol='carrier', outputCol='carrier_idx')

# Indexer identifies categories in the data
indexer_model = indexer.fit(flights)

# Indexer creates a new column with  index values
flights_indexed = indexer_model.transform(flights)

# Repeat the process for the other categorical feature
flights_indexed = StringIndexer(inputCol='org', outputCol='org_idx').fit(flights_indexed).transform(flights_indexed)
```

#### Assembling columns
Use a vector assembler to transform the data
🚨🚨 ==Machine Learning== algorithms in Spark operate on a single vector of predictors => A single column  `features`

```python
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(
	inputCols=['cyl', 'size'],
	outputCol = 'features'
 )

assember.transform(cars)
```

```ouput
+---+----+---------+
|cyl|size| features|+---+----+---------+|  3| 1.0|[3.0,1.0]||  4| 1.3|[4.0,1.3]||  3| 1.3|[3.0,1.3]|+---+----+---------+

```

### Decision Tree

- A [[Decision Tree]] is constructed by ==decision partitioning==

#### Split train/test

```python
# Perform train_test split
cars_train, cars_test = cars.randomSplit([0.8, 0.2], seed=233)
```

#### Build the Decision Tree

##### Training
```python
from pyspark.ml.classification import DecisionTreeClassifier

# Create the classifier
tree = DecisionTreeClassifier()

# Learn from the training data
 tree_model = tree.fit(cars_train)
```

##### Evaluation
- using the `.transform()` method.
```python
predictions = tree_model.transform(cars_test)
```

#### [[Confusion Matrix]]
- Compute a confusion matrix with Spark
```python
# Create a confusion matrix
prediction.groupBy('label', 'prediction').count().show()

# Calculate the elements of the confusion matrix
TN = prediction.filter('prediction = 0 AND label = prediction').count()
TP = prediction.filter('prediction = 1 AND label = prediction').count()
FN = prediction.filter('prediction = 0 AND label != prediction').count()
FP = prediction.filter('prediction = 1 AND label != prediction').count()

# Accuracy measures the proportion of correct predictions
accuracy = (TN + TP) / (TN + TP + FP + FN) 
print(accuracy)
```


### Logistic Regression
```python
from pyspark.ml.classification import LogisticRegression 

# Create the classifier
logistic = LogisticRegression()
# Learn from the training data
logistic = tree.fit(cars_train)
```

#### Precision and Recall
```python
# Precision (positive)
precision = TP / ( TP + FP )

# Recall
recall = TP / ( TP + FN )
```


#### Weighted metrics
Available metrics:
- `accuracy`
- `f1`
- `weightedPrecision`
- `weightedRecall`

```python
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

# Calculate precision and recall
precision = TP / (TP + FP)
recall = TP / (TP + FN)
print('precision = {:.2f}\nrecall = {:.2f}'\
      .format(precision, recall))

# Find weighted precision
multi_evaluator = MulticlassClassificationEvaluator()
weighted_precision = multi_evaluator.evaluate(
    prediction, {multi_evaluator.metricName: 
    "weightedPrecision"})

# Find AUC
binary_evaluator = BinaryClassificationEvaluator()
auc = binary_evaluator.evaluate(
    prediction, 
    {binary_evaluator.metricName: "areaUnderROC"})
```


### Turning text into tables
- **One record per document** => Corpus
- -> **One document** -> **Many columns** => #BOW-nlp 

#### Prepare the text
#NLP 
#nlp-preprocessing

##### Regular Expressions
- Regular Expression => #regex 

```python
from pyspark.sql.functions import regexp_replace

# REgular expression to match commas and hyphens
REGEX = '[,\\-]'

books = books.withColumn(
		'text', # Column to process
		regexp_replace(books.text, REGEX, ' ')
)
```

##### Text to Tokens

- `Tokenizer` -> will create a new column

```python
from pyspark.ml.feature import Tokenizer

books = Tokenizer(inputCol = 'text',
				 outputCol = 'tokens').\
		transform(books)

```

##### Stop words
```python
from pyspark.ml.feature import StopWordsRemover

stopwords = StopWordsRemover()

# view stopwords
stopwords.getStopWords()

# Remove stopwords from tokens columns
stopwords = stopwords.setInputCol('tokes').\
			setOutputCol('words')

books = stopwords.transform(books)
```

##### Feature hashing
- Converts words into numbers using hashes:
- It will create two lists inside each element:
	- The words representing by numbers
	- The frequency of each word

```python
from pyspark.ml.feature import HashingTf

hasher = HashingTF(inputCol='words',
				  outputcol = 'hash',
				  numFeatures = 32	
				  )
books = hasher.transform(books)
```

##### TF-IDF
#tf-idf 
- Dealing with common words at each document
- => Inverse document frequency
 
```python
from pyspark.ml.feature import IDF

# Compute the inverse document frequency
books = IDF(inputCol = 'hash',
		   outputCol = 'features').\
		fit(books).transform(books)
```


## Regression

### One-hot Encoding

==**Dummy variables**==

Data => sparse matrix.
![[Captura de Pantalla 2021-03-26 a la(s) 17.53.40.png]]

```python
from pyspark.ml.feature import OneHotEncoderEstimator

onehot = OneHotEncoderEstimator(
	inputCols = ['type_idx'], 
	outputCols = ['type_dummy'])

# Fit the encoder to the data
onehot = onehot.fit(cars)
# how many category levels?
onehot.categorySizes()

# Apply the transform to the data
cars = onehot.transform(cars)

# Explore the results
cars.select('type', 'type_idx', 'type_dummy').\
	distinct().\
	sort('type_idx').\
	show()
```

#### Dense versus sparse
#sparse-matrix 

### Regression

#### Loss function
#MSE => ==Mean Squared Error==

$$MSE = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y_i}^2)$$
- Minimize the average residual => Average distance between the observed and predicted values.

```python
from pyspark.ml.regression import LinearRegression

regression = LinearRegression(labelCol = 'consumption')

# Fit the model
regression = regression.fit(cars_train)

# Make the prediction
predictions = regression.transform(cars_test)
```

#### Calculate RMSE
#RMSE => Estandard deviation of the residuals.

```python
from pyspark.ml.evaluation import RegressionEvaluator

RegressionEvaluator(labelCol = 'consumption').evaluate(predictions)
```

#### Regression examination

```python
regression.intecept
regression.coefficients
```

### Bucketing & Engineering

- Convert a continuous value into a discrete value => bins or buckets

#### Buckets
```python
from pyspark.ml.feature import Bucketizer, \
						OneHotEncoderEstimator

# Create buckets at 3 hour intervals through the day
buckets = Bucketizer(
	splits=[0, 3, 6, 9, 12, 15, 18, 21, 24], 
	 inputCol='depart', 
	 outputCol = 'depart_bucket')

# Bucket the departure times
bucketed = buckets.transform(flights)
bucketed.select('depart', 'depart_bucket').show(5)

# Create a one-hot encoder
onehot = OneHotEncoderEstimator(
	inputCols = ['depart_bucket'], 
	outputCols=['depart_dummy'])

# One-hot encode the bucketed departure times
flights_onehot = onehot.fit(bucketed).transform(bucketed)
flights_onehot.select('depart', 'depart_bucket', 'depart_dummy').show(5)
```

### Regularization

#### #Feature-selection => Penalized Regression
Select only the most useful features.

$$MSE = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y_i}^2) + \lambda f(\beta)$$
- The regularization term depends on the coefficients.
- With a larger $\lambda$ more regularization 

There are two standard form of Regularization:
- ==Lasso== => absolute value of the coefficients ---> Forced to Zero
- ==Ridge== => square of the coefficients ---> Close to Zero

##### Ridge Regression

```python
ridge = LinearRegression(labelCol='consumption',
						elasticNetParam = 0, # Ridge Regression
						regParam = 0.1)
```

##### Lasso Regression

```python
ridge = LinearRegression(labelCol='consumption',
						elasticNetParam = 1, # Lasso Regression
						regParam = 0.1)
```

## Ensembles and Pipelines

### Pipelines
![[Captura de Pantalla 2021-03-26 a la(s) 19.17.35.png]]

#### Without a pipeline
The common process consists of:

```python
# Convert categorical strings to index values
indexer = StringIndexer(
	inputCol='org', 
	outputCol='org_idx')

# One-hot encode index values
onehot = OneHotEncoderEstimator(
    inputCols=['org_idx', 'dow'],
    outputCols=['org_dummy', 'dow_dummy']
)

# Assemble predictors into a single column
assembler = VectorAssembler(
	inputCols=['km', 'org_dummy', 'dow_dummy'], 
	outputCol='features')

# A linear regression object
regression = LinearRegression(labelCol='duration')
```


#### With a pipeline
##### The effective way of using `fit` and `transform`
![[Captura de Pantalla 2021-03-26 a la(s) 19.19.33.png]]

##### A pipeline is a better approach to keep things simple

Use the ==pipeline== to combine the multiple stages of the ML process.

```python
from pyspark.ml import Pipeline

# use the steps of the pipeline
pipeline = Pipeline(stages = [indexer, onehot, assemble, regression])

# PARSE the TRANING data
pipeline = pipeline.fit(cars_train)

# PARSE the TEST data
pipeline = pipeline.transform(cars_test)

# Access to each stage inside the pipeline
pipeline.stages # <= it will retun a list of stages
```

Another example:

```python
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF

# Break text into tokens at non-word characters
tokenizer = Tokenizer(inputCol='text', 
					  outputCol='words')

# Remove stop words
remover = StopWordsRemover(inputCol='words', 
						   outputCol='terms')

# Apply the hashing trick and transform to TF-IDF
hasher = HashingTF(inputCol='terms', 
				   outputCol="hash")
idf = IDF(inputCol='hash', 
		  outputCol="features")

# Create a logistic regression object and add everything to a pipeline
logistic = LogisticRegression()
pipeline = Pipeline(stages=
			[tokenizer, remover, hasher, idf, logistic])
```

### Cross-Validation
#cross-validation

```python
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

params = ParamGridBuilder().build()

cv = CrossValidator(estimator=regression,
				   estimatorParamMaps = params,
				   evaluator = evaluator,
				   numFolds =10)
```

Cross-validation along a Pipeline
```python
# Create a pipeline and cross-validator.
pipeline = Pipeline(stages=[indexer, onehot, assembler, regression])
cv = CrossValidator(estimator=pipeline,
          estimatorParamMaps=params,
          evaluator=evaluator)
```

### Grid-Search
#Grid-search

![[Captura de Pantalla 2021-03-26 a la(s) 19.46.27.png]]

#### Parameter grid
```python
from pyspark.ml.tuning import ParamGridBuilder 

# Create a parameter grid builder 
params = ParamGridBuilder()

# Add grid points
params = params.addGrid(regression.fitIntercept, [True, False])

# construct the grid
params = params.build()
```

#### Grid Search with Cross-Validation

```python
cv.avgMetrics

# The best model => the cv by default will behave as the best
cv.bestModel

# The best parameters
cv.bestModel.explainParam('parameter')
```

### Ensembles
==Wisdom of the Crowd== +> Collective opinion of a group better than that of a single expert.

##### Diversity and independence
> **Diversity** and **independence** are important because the best collective decisions are the product of disagreement and contest, not consensus or compromise
> - James Surowiecki -> *The wisdom of Crowds*

#### Random Forest

Bootstrap aggregation using Decision Trees

```python
# Create a random forest classifier
forest = RandomForestClassifier()

# Create a parameter grid
params = ParamGridBuilder() \
            .addGrid(forest.featureSubsetStrategy, 
					 ['all', 'onethird', 'sqrt', 'log2']) \
            .addGrid(forest.maxDepth, [2, 5, 10]) \
            .build()

# Create a binary classification evaluator
evaluator = BinaryClassificationEvaluator()

# Create a cross-validator
cv = CrossValidator(
    estimator = forest, 
    estimatorParamMaps = params, 
    evaluator = evaluator, 
    numFolds = 5)

# Feature importances
forest.featureImportances
```

#### Gradient-Boosted Trees

Iterative boosting algorithm -> Sequential fitting over the residuals of the previous trees.

```python
from puspark.ml.classification import GBTClassifier

gbt = GBTClassifier(maxiter = 10)

# Feature importances
bgt.featureImportances
```