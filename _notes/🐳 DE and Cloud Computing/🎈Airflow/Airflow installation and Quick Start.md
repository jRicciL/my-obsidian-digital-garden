# Airflow Installation and Quick start
#Airflow 

# Apache Airflow

- Apache Airflow is open-source for developing, scheduling and monitoring batch-oriented workflows.
	- Batch workflow orchestration
- A web interface helps manage the state of the workflows.
- Characteristics:
	- Dynamic: Airflow pipelines are configured as Python code → Dynamic pipeline generation
	- Extensible: Airflow framework contains operators to connect with numerous technologies.
		- All airflow components are extensible to easily adjust to your environment.
	- Flexible:
		- Workflow parameterization is built-in leveraging the Jinja templeting engine.

## Why Airflow?
- Batch workflow orchestration
- Contains operators to connect with many technologies and facilities to connect with new technologies.
- Workflows are programmed as Airflow DAGs.
- Coding over clicking:
	- Workflows can be stored in version control
	- Workflows can be developed by multiple people simultaneously
	- Test can be written to validate functionality
	- Components are extensible
	- User interfase → Pipelines and Tasks
- Airflow was not designed for event-based workflows.
	- It is not a streaming solution.
	- But can work together with Apache Kafka

## Instalation and quick start

This is the first approach to test Airflow locally. It uses the `standalone` command and relies on a `SQLite` database, which are not recommend for production environments.

<div class="rich-link-card-container"><a class="rich-link-card" href="https://airflow.apache.org/docs/apache-airflow/stable/installation/installing-from-pypi.html" target="_blank">
	<div class="rich-link-image-container">
		<div class="rich-link-image" style="background-image: url('https://airflow.apache.org/docs/apache-airflow/stable/_static/pin_32.png')">
	</div>
	</div>
	<div class="rich-link-card-text">
		<h1 class="rich-link-card-title">Installation from PyPI — Airflow Documentation</h1>
		<p class="rich-link-card-description">
		
		</p>
		<p class="rich-link-href">
		https://airflow.apache.org/docs/apache-airflow/stable/installation/installing-from-pypi.html
		</p>
	</div>
</a></div>

- My own installation
```bash
conda create -n airflow pip setuptools python=3.6
conda activate airflow
pip install "apache-airflow[s3, postgres]"

pip check
```

2. Set AirflowHome
```bash
export AIRFLOW_HOME=~/airflow
```

3. Run Airflow Standalone:
```bash
airflow standalone
```

4. See the Airflow generated password for the `admin` user here
```shell
vi $AIRFLOW_HOME/standalone_admin_password.txt
```

5. Go to http://localhost:8080/home
6. The installation creates the `airflow.cfg`
```bash
vi $AIRFLOW_HOME/airflow.cfg
```


### Alternative to `standalone`

```bash
airflow db init # For initializations
airflow db migrate

airflow users create \
    --username admin \
    --firstname Peter \
    --lastname Parker \
    --role Admin \
    --email spiderman@superhero.org

airflow webserver --port 8080

airflow scheduler
```

### Advanced → Production Deployment

https://airflow.apache.org/docs/apache-airflow/stable/administration-and-deployment/production-deployment.html

