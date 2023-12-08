  
# Introduction to DataBricks

<div class="rich-link-card-container"><a class="rich-link-card" href="https://campus.datacamp.com/courses/introduction-to-databricks/introduction-to-databricks?ex=1" target="_blank">
	<div class="rich-link-image-container">
		<div class="rich-link-image" style="background-image: url('https://campus.datacamp.com/public/assets/images/var/twitter_share.png')">
	</div>
	</div>
	<div class="rich-link-card-text">
		<h1 class="rich-link-card-title">Introduction to the Databricks Lakehouse Platform | Databricks</h1>
		<p class="rich-link-card-description">
		Here is an example of Introduction to the Databricks Lakehouse Platform: .
		</p>
		<p class="rich-link-href">
		https://campus.datacamp.com/courses/introduction-to-databricks/introduction-to-databricks?ex=1
		</p>
	</div>
</a></div>


# P1. Introduction to Databricks Lakehouse
- Databricks is a popular choice for enterprises
### Data Warehouse [[Lakehouse Fundamentals#Data warehouse 🏘️]]

**Pros:**
- Great for structured data
- Highly performant
- Predeffined schemas
- Easy to keep data clean
**Cons:**
- Very expensive
- Cannot support modern applications
- Not built for Machine Learning

### Data Lake
Basada en objetos no estructurados en la nube

**Pros**:
- Support for all use cases
- Very flexible
- Cost effective
**Cons**:
- Data can become messy
- Not very performant

## 🌊The Data Lakehouse
![[Captura de Pantalla 2023-11-20 a la(s) 7.02.59.png]]
- Single platform for all data workloads 
- Built on open source technology
- Colllaborative environment
- Simplified architecture
- Data is ==stored in an open-source format== for extensibility across teams

### Databricks Architecture Benefits
- Unification
	- Use cases ranging from Artificial Intelligence to Bussines Intelligence
	- Benefits of data warehouse and data lake
- Multi-Cloud
	- Bring powerful platform to the data
	- No lock-in to a specific cloud platform
- Collaborative
	- Every data persona is capable to work in the same platform in real time
 -  Open Source
	 - Underpinned by Apache Spark
	 - Support for different languages: 
		 - Python, R, Scale, SQL

## Core Features of the Databricks Lakehouse

### Apache Spark -> [[C2. Big Data Fundamentals with PySpark]]
![[C2. Big Data Fundamentals with PySpark#^3bb305]]

- Benefits of Spark
	1. Extensible, flexible open source framework
	2. Large developer community
	3. High performing
	4. Databricks optimizations for Spark

### Cloud computing assets
Grandes servidores a tu disposición para escalar tareas computacionales
- ==**Clusters**==
	- Collection of computational resources 
	- All workloads, any use case
	- All-purpose vs. Jobs
- **==SQL Warehouses
	- SQL only for BI use cases
	- Photon -> Aumenta significativamente el desempeño de SQL [[Lakehouse Fundamentals#Photon]]
- ==**Could data storage**==
	- DELTA [[Lakehouse Fundamentals#Delta Tables]]
		- Delta is an open-source data storage file format build around .parquet
		- Pros:
			- ACID transactions
			- Unified batch and streaming
			- Schema evolution -> Flexibility of the lake
			- Time travel

### Unity Catalog
> **Unity catalog** is an open data governance starategy that ==controls access== to all data asset in the Databricks Lakehous platform

Personas: 
- Data Scientists
- Data Engineer
- SQL Analyst

### Databricks UI
- All users have access to data and compute
- SQL users get a familiar interface for queries and reports
- Data engineers leverage Delta Live Tables
- Machine Learning workloads use models, features, and more

## Admnistering a Databricks Workspace

### Account Admin
- Full control
- Checks that all available environments are set up correctly
- **Key responsabilities:**
	- Creating and managing workspaces
	- Enabling Unity Catalog
	- Managing identities
	- Managing the account subscription
- Access to the account console:
	- List of worskpaces
	- Location of the data in the lake
	- Account usage in terms of DataBricks units (DBU)

### Workspace Admin
![[Captura de Pantalla 2023-11-20 a la(s) 7.35.19.png]]
**Key Responsabilities:**
- Managing identities in the workspace
- Creating and managing compute resources
- Managing workspace features and settings

### Databricks Workspace
Each cloud will have the same general options to create a workspace:
- Cloud service provider marketplace
- Account console
- Accounts APIs
- Programmatic deployments such as Terraform

- ==Data Plane== -> Third party Cloud Provider
	- Contains all of the customer's assets needed for computation with Databricks
	- Data is stored in the customer's cloud environment
	- Clusters/SQL warehouses run in customer's cloud tenant.
	- Databricks does not store users data.
- ==**Control Plane**== -> Managed by Databricks
	- The portion of the platform that is managed and hosted by Databricks
	- Orchestrates various backgrounds in databricks
	- Security and version updates
	- The web application resides here.
	- Responsible for tastks such as launching clusters and initiating jobs.
![[Captura de Pantalla 2023-11-20 a la(s) 8.09.22.png]]
# P2. Data Engineering
### Cluster creation
- Configuration options:
	- Cluster policies and access
	- Cluster Access
	- Types of clusters
		- Single user clusters
		- Shared clusters -> Several users, teams working on the same project
	- Databricks runtime
	- Photon acceleration
	- Physical properties
		- node instance types and number of nodes
		- Auto-scaling / Auto-termination
### Data Explorer
- Browse available catalogs/schemas/tables
- Look at sample data and summary statistics
- View data lineage and history

### Create a Notebook
- Standard interface for Databricks
- Support many languages
- Built-in visualizations
- Realtime collaboration
## ==Medallion architecture==
[[Medallion Architecture]]
![[Pasted image 20231120082842.png]]
1. 🟤 <mark>Bronze</mark>: “Landing zone” → Raw data
2. 🔵 ==Silver==: 
	- Clean, Transformed and join together
	- Ready for the majority of the data analysis applications
3. 🟡 Gold:
	- Data aggregated for BI reports and dashboards
## Reading Data

- Delta tables
- file formats
- Databases
- Streaming data
- Binary data

```python
# Delta table
spark.read.table()

# CSV files
spark.read.format('csv').load('file.csv')

# Postgres table
spark.read.format("jdbc")
	.option("driver", driver)
	.option("url", url)
	.option("dbtable", table)
	.option("user", user)
	.option("password", user)
	.load()
```

## Delta format

A delta table provides table-like qualities to an open file format.
- Feels like a table when reading
- Access to underlying files (parquet and json)

**Structure**
- ==_delta_log_== a JSON transaction log:
- parquet files that contain the data.

## DataFrames
- Dataframes are two-dimensional representations of data.
	- Look and feel similar to tables
	- Underlying construct for most data processes

## Writing data
- **Kinds of tables in Databricks**

1. ==Managed tables==:
	- Default type
	- Stored with Unity Catalog
	- Databricks managed
	- do not require additional configuration
1. Unmanaged tables:
	1. Stored in another location → A datalake of the cloud provider
	2. Are customer managed
	3. We need a `.location()` parameter

- Creating an EXTERNAL TABLE using SQL
```sql
CREATE TABLE example_table
USING postgresql

OPTIONS(
	dbtable 'table-name'
	host 'host-url'
	port '5432'
	database 'database-name'
	usser 'username'
	password 'password'
)
LOCATION <path-to-data-lake>
```


## Data Ingestion
### [[AutoLoader Databricks]]
Auto Loader precesses new data files as they land in a data lake.
- Incremental processing
- Efficient processing
- Automatic

```python
spark.readStream
	.format("cloudFiles")
	.option("cloudFiles.format", "json")
	.load(file_path)
```

### Structured Streaming
Read directly from streaming data
Process:
- Gather information about Kafka topic
- Read the data into a Structured Streaming Data Frame
- Clean the streaming data
- Join our other datasets into the stream
- Write the stream as a Delta table in a new data lake bucket
#Kafka
![[Pasted image 20231120121243.png]]
## Data Transformations
### Schema manipulation
- Add an remove columns
- Redifine columns

```python
df.withColumn(col('newCol'), ...)
  .drop(col('oldCol'))
```

### Filterning

```python
df.filter(F.col('date') >= target_date)
  .filter(F.col('date').isNotNull())
  .show()
```

### Aggregation
```python
df.groupBy(col('region'))
	.agg(sum(col('sales'))) 
```


## Orchestration in Databricks

- Automate and orchestrate the transformatiosn
	- Orchestration → Atomation
	- End-to-end data life cycle → Ingestion to serving
## [[Databricks Workflows]]

- Databricks workflows is a collection of built-in capabilities to orchestrate all your data processes at no additional cost.
- What can we orchestrate?
	- **Data engineers:** Databricks Notebooks, Delta live tables
	- **Data analysts:** Queries, dashboards and alerts
	![[Captura de Pantalla 2023-11-21 a la(s) 23.37.56.png]]
### Databricks Jobs^[[Databricks Workflows#Databricks Jobs]]

- **Jobs can be crated directly from the UI**:
	- Directly from a notebook
	- In the workflows section
- **Programmatically**:
	- Command line tool
	- Jobs API

### [[Delta Live Tables]]
#DeltaLiveTables
- A declarative framework for building reliable, maintainable, and testable data processing pipelines.
	- You define the transformations to perform on your data and Delta Live Tables manages:
		- Task orchestration
		- Cluster management
		- Monitoring
		- Data quality
		- Error handling
- You define streaming tables and materialized views that the system should crate and keep up to date.
	- Instead of defining your data 

# C3 - SQL and Warehousing

## Databricks SQL
- Data Warehousing for the Lakehouse
- Familiar environment for SQL users
- Optimized with the Photon engine
- Connect to BI tools
- Typically used on the “Gold” layer

Differences with other Warehouses
- Open file format (Delta)
	- Other data process can access same data
- Separates the compute power from storage
- Based on ANSI SQL and Spark
- Can be integrated into other data workloads
	- Advanced analytics such as Machine Learning

### Creating a SQL Warehouse

- Require a SQL Warehouse → Optimized for SQL
- Configuring a SQL Warehouse
	- Cluster name
	- Cluster Size
	- Scaling behaviour
	- Cluster type
		- Classic: 
			- Most basic SQL compute
			- Exists in customer cloud
		- Pro:
			- More advanced features
			- In customer cloud
		- Serverless:
			- Cutting edge features
			- Exists in Databricks cloud
			- Most cost performant

## Databricks SQL queries and dashboards
### Visualizations
- Lightweight, in-platform visualizations
- Support for standard visual types
- Ability to quickly comprehend data in a graphical way

# C3 - Large scale applications and Machine Learning

### Why the Lakehouse for AI / ML?
- Reliable data and files in the Delta lake
- Highly scalable computing
- Open standards, libraries, frameworks
- Unification with other data teams
## MLOps  in the Lakehouse
#MLOps

![[Captura de Pantalla 2023-11-29 a la(s) 19.32.34.png]]
### DataOps
- Integrating data across different sourses using #AutoLoader → [[AutoLoader Databricks]]
- Transforming data into a usable, clean format → #DeltaLiveTables  → [[Delta Live Tables]]
- Creating useful features for models → #FeatureStore
![[Captura de Pantalla 2023-11-29 a la(s) 19.40.02.png]]
### ModelOps
- Develop and train different models → Databricks Notebooks
- Machine learning templates and automation → Generate a baseline model → #AutoML
- Track parameters, metrics, and trials → #MLFlow framework
- Centralize and consume models → #ModelRegistry
![[Captura de Pantalla 2023-11-29 a la(s) 19.40.12.png]]
### DevOps → Production
- Govern access to different models → #UnityCatalog
- Continuos Integration and Continuos Deployment (CI/CD) for model versions → #ModelRegistry 
- Deploy models for consumption → #ServingEndpoints
![[Captura de Pantalla 2023-11-29 a la(s) 19.40.25.png]]
## Planing for Machine Learning

![[Captura de Pantalla 2023-11-29 a la(s) 19.43.15.png]]
- What do we have?
	- Data availability
	- Business requirements
	- Data scientists / Data analysts 
- What do I want?
	- Use cases
	- Legal and security compliance
	- Business outcomes

## ML Runtime
- Extension of Databricks compute
- Optimized for machine learning applications
- Contains most common libraries and frameworks
	- `scikit-learn` `SaparkML` `TensorFlow`
	- `MLFlow`
- Works with cluster library management

## Exploratory Data Analysis
#EDA #ExploratoryDataAnalysis #python 
```python
import bmboolib as bam
# For UI exploratory data analysis
```

## Feature tables and feature stores
![[Captura de Pantalla 2023-11-29 a la(s) 19.54.12.png]]
### Databricks Feature Store

- Centralized storage for featurized datasets
- Easily discover and reuse features for machine learning models
- Upstream and downstream lineage

Using the feature store
```python
from databrikcs import feature_store


df = spark.read.table("review_data")
dbutils.data.summarize(df)

feature_df = (
	df.withColumn('feature1', ...)
	  .withColumn('feature2', ...)
)


fs = feature_store.FeatureStoreClient()

fs.create_table(
	name = table_name,
	primary_keys = ['wine_id'],
	df = features_df,
	schema = features_df.schema,
	description = "wine features"
)
```

## Model Traning with AutoML and MLFlow in Databricks
Model Develpment and Model Evaluation
#AutoML #MLFlow 

### AutoML
- Glass box approach to automated machine learning
	- Great for “Citizen data scientist”
- Uses the most popular libraries
- Can create a model that best predicts what you are aiming to predict 
- UI based
- Provides notebook with generated code to reproduce the model

### MLFlow
- Open source framework
- Ent-to-end machine learning lifecycle management
- Track, evaluate, manage, and deploy
- Preinstalled on ML Runtime

Procedure:
- Define an experiment
- Start the machine learning run
- Compare your model runs in the MLFlow experiment
- Select the model from the best run

```python
import mlflow

# Start the machine learning run
with mlflow.start_run() as run:
	# Machine leraning training

# Train your model and track your information
mlflow.autolog()
mlflow.log_metric('accuracy', acc)
mlflow.lot_param('k', kNum)
```

### MLFlow Experiments
- Collect information across multiple runs in a single location
- Sort and compare model runs → Based on different metrics

## Deploying a model in Databricks
- Get sure that high quality models get into production
![[Pasted image 20231130073606.png]]

Concerns with deploying models:
1. **==Availability==**
	- How will the ML model be used?
	- Where do I need to put my model to access it?
	- Will the model be easy to understand or use?
2. **==Evaluation==**
	- Are my users actually using my model?
	- Is my model still performing well?
	- Do I need to retrain my model?
	- Do I need a new model that is better?

### Model Flavors → Model Registry
#ModelFlavors

**Model Falvors**
- MLFLow Models can store a model from any machine learning framework
- Models are stored alongside different configurations and artifacts
- Models can be “translated” into another kind of model based on needs.
	- Different ==flavors==:
		- `scikit-learn`
		- `pyfunc`
		- `spark`
		- `tensorflow`

**Model Registry:**
- Is a collection of all models we have train
- Including previous versions of these models → Version History
	- Easily see current and past version of the same model
- From here we can push a model into a **production** or **staging**

### Model Serving → Deployment 🚀
- From local models to docker containers
- Model Serving comes with a dashboard to measure performance
- Advantages:
	- Simplify the overhead of managing compute resources and provide built-in capabilities for model monitoring.

## Simple End-to-end ML pipeline
1. Transform datasets, create features, and store them in the Feature Store
2. Train your model using the SparkML library
3. Test out your model against another dataset and track the results with MLFlow
4. Store your trained and tested model in the Model Registry
5. Server your model with a Databricks serving endpoint for your end users to consume

