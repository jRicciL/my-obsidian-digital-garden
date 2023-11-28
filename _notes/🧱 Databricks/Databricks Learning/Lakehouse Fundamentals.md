# Learning objectives and contents

- Learning objectives
    
    **Module 1: What is a Data Lakehouse?**
    
    - **Describe the origin and purpose of the data lakehouse.**
    - Explain the challenges of managing and using big data.
    
    **Module 2: What is the Databricks Lakehouse Platform?**
    
    - Describe fundamental concepts about the Databricks Lakehouse Platform.
    - Give examples of how the Databricks Lakehouse Platform solves big data challenges.
    - Explain how the Databricks Lakehouse Platform benefits Data Engineers, Data Analysts, and Data Scientists.
    
    **Module 3: Databricks Lakehouse Platform Architecture and Security Fundamentals**
    
    - Describe essential platform components and features essential to data reliability, performance and governance (Delta Lake, Photon, Unity Catalog, Delta Sharing, serverless compute).
    - Define essential platform terminology (metastore, catalog, schema, table, view, and function).
    - Describe compute resources for your Databricks Lakehouse Platform.
    
    **Module 4: Supported Workloads on the Databricks Lakehouse Platform**
    
    - **Explain how the Databricks Lakehouse Platform supports the following workloads: data warehousing, data engineering, data streaming, and data science and machine learning.**
    - Describe the benefits of using the Databricks Lakehouse Platform for **data warehousing, data engineering, data streaming, and data science and machine learning.**

# What is a data Lakehouse?

- [https://customer-academy.databricks.com/learn/course/1325/play/8329/what-is-a-data-lakehouse;lp=215](https://customer-academy.databricks.com/learn/course/1325/play/8329/what-is-a-data-lakehouse;lp=215)

### The history of data management in analytics

- 1980 business neeed more than relational databases
- High volumes of data

## Data warehouse 🏘️

A data warehouse is a centralized repository that stores historical data from operational databases and other sources. It is designed to facilitate data analysis and reporting.

🟡 Pros:

- Business intelligence (BI)
- Analytics
- Structured and clean data
- Predeffied schemas

🔴 Cons:

- No support for semi or unstructured data
- Inflexible schemas
- Struggled with volume and velocity upticks
- Long processing time

## DataLakes 🌊

A data lake is a scalable repository that holds a vast amount of raw data in its native format until it is needed. Data lakes allow you to store data without having to define its structure first. It is designed to store all your data at any scale.

Pros:

- Flexible data storage
    - Data created from many different sources
- Streaming support
- Cost efficient in the cloud
- Support for AI and Machine Learning

Cons:

- Not supportive of transactional data
    - Transactional data refers to data that changes regularly and frequently and is crucial for day-to-day operations.
- Poor data reliability
- Slow analysis performance
- Data governance concerns
- Data warehouses still need

## Two incompatible platforms?

- The main differences between a data lake and a data warehouse are:
    
    |Data Warehouse|Data Lake|
    |---|---|
    |Cleansed and structured data|Raw, unstructured data in its native format|
    |No predefined schema|Rigid schema|
    |Optimized for analysis and reporting:|Optimized for huge amounts of data|
    |BI and SQL analytics|Data Science and ML, Data Streaming|
    |Ready for analysis|Modeling before analysis|
    |More time and resources to maintain|Cheaper to build and scale|
    

![Captura de Pantalla 2023-02-28 a la(s) 8.51.46.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/cb2101db-04d1-4f27-9e79-e552b488bb90/Captura_de_Pantalla_2023-02-28_a_la(s)_8.51.46.png)

## Data Lakehouse

A data lakehouse is a modern data architecture that combines the best of data lakes and data warehouses.

- It is an enterprise data management system that delivers the **flexibility** of a data lake and the **governance and performance** of a data warehouse.
- Open architecture

![Captura de Pantalla 2023-02-28 a la(s) 8.57.17.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/231d599b-60b9-4e71-b5c0-5aee6a71bbeb/Captura_de_Pantalla_2023-02-28_a_la(s)_8.57.17.png)

- Built on a data lake, a data lakehouse can store all data of any type together

### Key Fetatures

- Transaction support
- Schema enforcement and governance
- Data governance
- BI support
    - Reduce the latency between obtaining data and getting insights
- Decoupled storage from compute
    - Each operate independently, allowing them to scale independently to support specific needs
- Open storage formats
    - Apache parquet ⇒
- Support for diverse data types
- Support for diverse workloads
    - Data science, machine learning, and sql analytics fro mthe same source

# DataBricks Lakehouse platform

Databricks is a unified data analytics platform founded in 2013 by the creators of Apache Spark. It provides a collaborative workspace for data engineers, data scientists, and business analysts. The Databricks Lakehouse Platform combines the capabilities of data warehouses and data lakes into a single, unified platform. The Databricks Lakehouse Platform enables users across the organization to work with data. It supports workflows such as:

- Data engineering: Ingesting and preparing data for analytics
- Data science: Training machine learning models
- Business intelligence: Creating reports and dashboards
- Data warehousing: Supporting enterprise data analytics Some key projects built on Databricks include:
- Delta Lake: Reliable data lake technology
- MLflow: Machine learning lifecycle management
- Koalas: pandas API on Apache Spark
- Apache Spark: Unified analytics engine for big data processing

## Welcome to Databricks

- **Lakehouse** ⇒ The name comes from the research paper “Lakehouse a new Generation of open platform that unify data warehousing and advanced analytics”
- Unifies warehousing and AI usecases
    - Data Engineering, Data Streaming, Data Science and ML
- Details about the Lakehouse paradigm:
    - One security and governance approach
    - Cloud agnostinc
    - A reliable data platform ⇒ manage all data types
- Realized on Databricks
    - Supports for Persona-based use cases
    - Unit Catalog:
        - Fine-grined governance for data and AI
    - DeltaLake
        - Data reliability and performance
    - Cloud Data Lake

### Databricks main features

- Simple:
    - Unify your data warehousing and AI use cases on a single platform
- Open:
    - Built on open souce and open standards
    - Allows open source projects
- Multicloud
    - One consistend data platform across clouds

# Architecture and Security Fundamentals

### Why is data reliability and performance important?

- Garbage in ⇒ garbage out
- Data lakes ⇒ Data Swamp
    - A "data swamp" refers to a poorly governed data lake that lacks data quality, reliability, and performance. It fails to provide value due to issues like data duplication, inconsistency, and lack of discoverability.

### Problems encountered when using data lakes

- Poor ACID transaction support:
    - Atomicity, Consistency, Isolation, and Durability
- Lack of schema enforcement
- LAck of integration with a data catalog
- Ineffective paritioning ⇒ Poor indexing, ineffective due to high cardinality columns
- Too many small files ⇒ Coause for query performance degradation

### Databricks ⇒ DeltaLake and Photon

- The issues above are solved mainly by two fundational technologies

## DeltaLake:

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c13a4563-9d94-418d-b1c0-bf835d57f99b/Untitled.png)

Runs on top of existing data lakes and is compatible with Apache Spark

- File-based open source storage format
- Provides guarantees for ACID transactions
    - ⇒ No partial or corrupted files
- Scalable data and metadata handling
    - Leveraging Spark to scale out all the metadata processing
    - PEtabyte scale tables
- Audit history and time travel
- Schema enforcement and schema evlution
- Support for deletes, updates and merges
- Unified streaming and batch data processing

### Delta Tables

- Delta Tables
    
    Delta tables are ACID and versioned tables that run on top of your existing data lakes. They provide:
    
    - **Scalable metadata handling.** Leverages Apache Spark to scale out metadata processing and enables petabyte-scale tables.
    - **Transactional guarantees.** Provides ACID (Atomicity, Consistency, Isolation, Durability) transactions, ensuring no corrupted data or partial writes.
    - **Time travel.** Allows accessing previous versions of data, enabling auditing and rollbacks.
    - **Schema enforcement and evolution.** Enforces schemas while allowing schema changes over time in a backward-compatible manner.
    - **Streaming and batch data processing.** Unifies streaming and batch data processing under a common framework.
    - Deletes and overwrites. Supports deleting and updating rows in the table.
- Based on Apache parquet
    
- Usable with semi-structured and unstructured data
    
    - Provide versioning, reliability, metadata managemente, tiem travel capabilities

## Photon

To support the Lakehouse paradigm, the execution engine has to provide the same performance while still have the scalability of a data lake

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d3fb9e49-4e44-4418-973f-1f0696238010/Untitled.png)

### Photon main features

Photon is Databricks’ query optimizer and execution engine. It provides:

- Optimized query plans across partitions and files
- Vectorized query execution for fast in-memory processing
- Cost-based optimizations for data shuffling and joins
- Adaptive query execution to dynamically adjust to data and cluster conditions
- Built-in caching for faster query responses
- Seamless integration of SQL and DataFrame/Dataset APIs.
- Compatible with Spark APIs
    - Increase speed in use cases

Photon allows for interactive SQL queries on petabytes of data in your data lakehouse.

### Reported workloads impacted by Photon

- SQL-based jobs
- IoT use cases
- Data privacy
- Loading data into Delta and Parquet

# Unified Governance and Security

- The Databricks Lakehouse Platform provides unified governance and security across the platform through:
    - **Role-based access control (RBAC):** Assigns permissions to users based on their role in the organization. Supports fine-grained control over data and resources.
    - **Attribute-based access control (ABAC):** Assigns permissions based on attributes of the user, resource, and context. Enables dynamic control over who can access what data.
    - **End-to-end encryption:** Encrypts data at rest and in transit to protect sensitive information.
    - **Audit logging:** Provides a detailed audit trail of user activity, API calls, and data access for compliance and monitoring.
    - **Private endpoints:** Restricts access to your Databricks workspace to only authorized resources within your virtual private cloud (VPC).
    - **IAM integration**: Integrates with Identity and Access Management (IAM) services to manage access to resources across services.
    - **Data masking**: Obfuscates sensitive data to minimize the risk of unauthorized data exposure. Allows sharing data with a wider range of users while protecting privacy.
    - **Data retention policies**: Allows setting data retention policies to automatically purge old or unnecessary data from your data lakehouse. Ensures you only keep data that is useful and compliant with regulations.

### Challenges to data and AI Governance

- Diversity of data and AI assets
- Using two disparate and incompatible data platforms
- Rise of multi-cloud adoption
- Fragmented tool usage for data governance

### Databricks Data Governance Solutions

1. Unity Catalog
2. Delta Sharing
3. Divided architecture in two planes
    1. Control plane
    2. Data plane

## Unit catalog

![Captura de Pantalla 2023-03-03 a la(s) 21.03.21.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/0506d7ec-1c0c-499d-acc2-533e115d645a/Captura_de_Pantalla_2023-03-03_a_la(s)_21.03.21.png)

- Unified governance solution for all data assets
    - Provides a common governance model based on ANSI SQL
    - Supplies one consistent model to discover, access and share data
    - There is a single source of truth for all user identities and data assets in the Databricks Lakehouse Platform
    - A single access point
    - Removes data team silos
    - Logs who has performed which action
- The Unity Catalog is Databricks’ data catalog which provides fine-grained data governance across your data Lakehouse. It helps you:
    - Discover datasets and organize them hierarchically
    - Track lineage and dependencies between datasets to understand data flow
    - Manage metadata such as schema, descriptions, owners and tags
    - Enforce data quality rules and monitor metrics
    - Set access control policies over datasets
    - Gain visibility into how datasets are used across your organization The Unity Catalog allows centralized and unified governance of all your data assets in the lakehouse. It helps ensure data quality, security, and compliance which are essential for data-driven organizations.

## Delta sharing

![Captura de Pantalla 2023-03-03 a la(s) 21.07.25.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/aa23b75c-9021-42d2-8bac-446120fbb66d/Captura_de_Pantalla_2023-03-03_a_la(s)_21.07.25.png)

Delta sharing is a scalable data sharing technology built by Databricks. It enables sharing petabytes of data across organizations in a secure and compliant manner.

- Tradditionaly data sharing technologies do not scale well
- Delta Sharing is a simple REST protocol that shares access to part of a cloud data set

### Benefits of Delta Sharing

- Open cross-platform sharing
    - Provides integration to mulitple sresources
- Share live data without copying it
- Centralized administration and governance
- Marketplace for data products
    - Built and package data products
- PRivacy-safe data clean rooms

## Divided security architecture

![Captura de Pantalla 2023-03-03 a la(s) 21.12.13.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b8c07081-8626-4e52-bdad-9d4d80cdef49/Captura_de_Pantalla_2023-03-03_a_la(s)_21.12.13.png)

1. Control plane
    1. Managed backed services that Databricks provides
    2. Databricks runs wht workspace application and manages notebooks, configuration and clusters
2. Data plane
    1. It is where the data is process
    2. The compute resources in the data plane run inside the business owners own cloud account
    3. All the data stays where it is
    4. The information is incripted at REST

### User identity and access

An IAM instance profile in Databricks is an AWS Identity and Access Management (IAM) role that is attached to an EC2 instance. It allows the instance to make AWS service calls using the permissions assigned to that role.

**Lakehouse** ⇒ The term "lakehouse" comes from the research paper “Lakehouse: A New Generation of Open Platform That Unifies Data Warehousing and Advanced Analytics.”

The types of clusters available in Databricks are:

- Standard: For general-purpose workloads. It provides balanced CPU and memory resources.
- Memory Optimized: For memory-intensive workloads. It provides a high amount of memory per CPU.
- CPU Optimized: For CPU-intensive workloads. It provides a high amount of CPUs per memory.
- GPU: For workloads that leverage graphics processing units (GPUs) for parallel processing. It provides GPUs in addition to CPUs.
- Autoscaling: Automatically scales computing resources up and down based on workload demand. It is ideal for workloads with variable resource needs.

The Databricks Lakehouse Platform unifies data warehousing and AI use cases into a single platform. Key benefits of using Databricks Workflows for orchestration purposes include:

- A single platform for data engineering, data science, and business intelligence workflows. This eliminates the need to switch between tools and interfaces.
- Support for the entire machine learning lifecycle from data ingestion and preparation to model training, deployment, and monitoring.
- Integration with Delta Lake for data reliability and performance. Delta Lake runs on top of data lakes and is compatible with Apache Spark, providing ACID transactions, scalable metadata handling, and time travel.
- Support for streaming and batch data processing in a unified framework.
- Fine-grained governance and security across the platform with features like role-based access control, attribute-based access control, end-to-end encryption, and audit logging.
- An optimized query engine called Photon that enables interactive SQL queries on petabytes of data. Photon provides optimized query plans, vectorized query execution, and built-in caching for faster query responses.
- Integration with various AWS services like S3, IAM, and KMS for cloud deployment and management.
- An open platform built on open-source technologies and open standards. It allows you to build on what you already have and scale as your needs grow.