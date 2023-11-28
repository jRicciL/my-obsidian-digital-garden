# 🥇 Medallion Architecture

#MedallionArchitecture #Multi-hop


<div class="rich-link-card-container"><a class="rich-link-card" href="https://www.databricks.com/glossary/medallion-architecture" target="_blank">
	<div class="rich-link-image-container">
		<div class="rich-link-image" style="background-image: url('https://www.databricks.com/static/og-databricks-58419d0d868b05ddb057830066961ebe.png')">
	</div>
	</div>
	<div class="rich-link-card-text">
		<h1 class="rich-link-card-title">What is a Medallion Architecture?</h1>
		<p class="rich-link-card-description">
		A medallion architecture is a data design pattern used to logically organize data in a lakehouse, with the goal of improving the structure and quality of data.
		</p>
		<p class="rich-link-href">
		https://www.databricks.com/glossary/medallion-architecture
		</p>
	</div>
</a></div>

![[Pasted image 20231120090316.png]]
## What is a medallion architecture?

> Is a *data design pattern* used to logically organize data in a `Lakehouse` in layers.
- Incrementally and progressively improving:
	- the structure
	- quality of data
- ==**Layers in the Lakehouse**==
	- 🥉 Bronze → 🥈Silver → 🥇Gold

### Delta live tables
- Allow users to build data pipelines with Bronze, Silver and Gold tables.

## 🥉 Bronze Layer → *Raw Data*
- Land all the data from external source systems.
- Surce table structures “as-is"
- Include metadata columns that capture the load date/time, process ID, etc.
- Provide historical archive:
	- of source (cloud storage)
	- data lineage
	- Auditability 
	- Reprocessing
## 🥈 Silver Layer → *Cleansed and conformed data*
- The data from the Bronze layer is:
	- matched, merged, conformed and cleansed (“just-enough”)
	- ELT is followed by EtL
	- Provides new “Enterprise view” of all its key business entities:
		- Concepts and transactions
		- Master customers
		- Stores
		- non-duplicated transactions
		- Cross-reference tables
- Use cases:
	- Self-served analytics for ad-hoc reporting
	- Advanced Analytics
	- Machine Learning
- Users:
	- Departmental Analysts
	- Data Engineers
	- Data Scientists

## 🥇 Gold Layer → *Curated business-level tables*
- Typically organized for consumption-ready “project-specific” databases.
- The final layer of data transformations and data qualiy rules are applied here.
- Used for reporting.
- Uses more de-normalized data models with fewer joins.
- Use cases:
	- Customer Analytics
	- Customer Segmentation
	- Product Recommendations
	- Sales Analytics
- Lot of ==**Kimball style**== star schema
- **==Inmon style==** Data marts 