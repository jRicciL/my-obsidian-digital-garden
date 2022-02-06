---
---

# Data Leakage

-> ==Leakage==: Describes the situation where the data using to training the ML algorithm includes, unexpected, extra information about the target value.
- Introducing any other information about the target during training that would not legitimately be available during actual use =>
	- Variables highly predictive of the target, but not legitimately available at the time prediction needs to be done.
-> Simplest example:
- Include the label as the predictive value.
- Including test data with training data.

-> Leakage can cause the system to learn a sub optimal model that does much worse in an actual deployment.

### Two main types of leakage

#### Leakage in training data:
- Performing data preprocessing using parameters or results from analyzing the entire dataset:
	- Normalizing and rescaling using the whole dataset.
	- Estimating missing values using the whole dataset.
	- Perform feature selection using the whole dataset.

#### Leakage in features
- Removing variables that are not legitimate without also removing variables that encode the same or related information.
- Revesting anonumization

## Minimizing data Leakage
1. Perform data preparation within each cross-validation fold ==separately==:
	- Scale/normalize data
	- Perform feature selection

2. With time series data, use a timestamp cutoff:
	- This will make sure you are  not accessing any data records that were gathered after the prediction time.

3. Before any work with a new dataset, split off a final test validation dataset
	- Only if you have enough data
