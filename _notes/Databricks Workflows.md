# Databricks Workflows

#Databricks #Workflows
<div class="rich-link-card-container"><a class="rich-link-card" href="https://docs.databricks.com/en/workflows/index.html" target="_blank">
	<div class="rich-link-image-container">
		<div class="rich-link-image" style="background-image: url('https://www.databricks.com/wp-content/uploads/2020/04/og-databricks.png')">
	</div>
	</div>
	<div class="rich-link-card-text">
		<h1 class="rich-link-card-title">Introduction to Databricks Workflows</h1>
		<p class="rich-link-card-description">
		Learn how to orchestrate data processing, machine learning, and data analysis workflows on the Databricks Data Intelligence Platform.
		</p>
		<p class="rich-link-href">
		https://docs.databricks.com/en/workflows/index.html
		</p>
	</div>
</a></div>

### Example
The following diagram illustrates a workflow that is orchestrated by a Databricks job to:
![[Pasted image 20231121233521.png]]
1. Run a Delta Live Tables pipeline that ingests raw clickstream data from cloud storage, cleans and prepares the data, sessionizes the data, and persists the final sessionized data set to Delta Lake.
    
2. Run a Delta Live Tables pipeline that ingests order data from cloud storage, cleans and transforms the data for processing, and persist the final data set to Delta Lake.
    
3. Join the order and sessionized clickstream data to create a new data set for analysis.
    
4. Extract features from the prepared data.
    
5. Perform tasks in parallel to persist the features and train a machine learning model.

## Databricks Jobs

#Jobs


<div class="rich-link-card-container"><a class="rich-link-card" href="https://docs.databricks.com/en/workflows/jobs/jobs-quickstart.html" target="_blank">
	<div class="rich-link-image-container">
		<div class="rich-link-image" style="background-image: url('https://www.databricks.com/wp-content/uploads/2020/04/og-databricks.png')">
	</div>
	</div>
	<div class="rich-link-card-text">
		<h1 class="rich-link-card-title">Create your first workflow with a Databricks job</h1>
		<p class="rich-link-card-description">
		Learn how to quickly create and orchestrate tasks with a Databricks job.
		</p>
		<p class="rich-link-href">
		https://docs.databricks.com/en/workflows/jobs/jobs-quickstart.html
		</p>
	</div>
</a></div>

