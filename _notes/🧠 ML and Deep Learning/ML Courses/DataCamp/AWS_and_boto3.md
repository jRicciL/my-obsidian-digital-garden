---
---

# Introduction to AWS and Boto3
#AWS #python #boto3

## Intro to AWS and Boto3

### Introduction

Boto3 -> AWS

What is AWS service?
- AWS services are granular

```python
import boto3

s3 = boto3.client('s3',
				 region_name='us-east-1',
				  aws_access_key=AWS_KEY_ID,
				  aws_secret_access_key=AWS_SECRET
				 )

response = s3.list_buckets()
print(response)
```

```
{'ResponseMetadata': {'HTTPStatusCode': 200, 'HTTPHeaders': {}, 'RetryAttempts': 0}, 'Buckets': \[{'Name': 'datacamp-hello', 'CreationDate': datetime.datetime(2006, 2, 3, 16, 45, 9, tzinfo=tzlocal())}, {'Name': 'datacamp-uploads', 'CreationDate': datetime.datetime(2006, 2, 3, 16, 45, 9, tzinfo=tzlocal())}, {'Name': 'datacamp-transfer', 'CreationDate': datetime.datetime(2006, 2, 3, 16, 45, 9, tzinfo=tzlocal())}\], 'Owner': {'DisplayName': 'webfile', 'ID': 'bcaf1ffd86f41161ca5fb16fd081034f'}}
```

#### AWS Console

We can use `boto3` to manage the console.
- We need to use ==IAM User== to create permissions to boto3.
	- Credentials

![[Captura de Pantalla 2021-02-18 a la(s) 21.28.36.png]]

#### AWS Services
- **IAM**: Identity and Access Management, controls the access to the AWS resources
- **S3**: Storage service
- **SNS**: Simple Notification Service
- **Comprehend**: Sentimental Analysis
- **Rekognition**: Text and Images
- **EC2**: Amazon Elastic Compute
- **Amazon Redshift**


### S3: Diving into buckets

#### S3 buckets

- Let us put any file in the cloud.
- Managing cloud storage is a key component

##### Buckets and Objects

- Buckets -> like folders 
	- HAve their own permission policy
	- Website storage
	- Generate logs about their activity
	- Contain objects
	- Operations:
		- Create
		- List
		- Delete

- Objects:
	- Images
	- CSV
	- Anything


#### Bucket creation
