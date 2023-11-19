# HPC - Working with the scheduler

- An HPC system might have thousands of nodes and thousands of users.
	- The scheduler manages which jobs run where and when

![[Pasted image 20220305202012.png]]

## Running a Batch job

- Any command that you want to run on the cluster is called a==job==
	- The process of using a schedule to run the job is called *batch job submission*


### A shell script as an example

- First line => `#!/usr/bin/env bash`
	- `#!` -> Known as `hash-bang` or `shebang` tells the computer what program is meant to process the contents of this file.
	- I we want to use ==python== instead we need to provide the path to the executable
		- `#!/usr/bin/python3`
- Anywhere below the first line we will add our commands
	- In this case is just an `echo`
- On the last line we will invoke the `hostname` command which will print the name of the machine the script is run on


```bash
#!/usr/bin/env bash

echo -n "This script is running on "
hostname
```

### Details about file permissions
- Go to [[Introduction to HPC]]
- Go here: https://carpentries-incubator.github.io/hpc-intro/13-scheduler/index.html


#### File permissions

```bash
chmod u+x example-job.sh
ls -l example-job.sh
```

#### Execute the script in `loging` and in `worker` nodes


- To run in the `loging` node

```bash
./example-job.sh
```
```
This script is running on gra-login1
```

- To run in a `worker` node => using ==sbatch==

```bash
[yourUsername@gra-login1 ~]$ sbatch example-job.sh
```

```
Submitted batch job 36855
```

## Customising a Job



  
**Feedback:** SSH protocols will encrypt every file which is copied therefore lots of files introduce a lot of additional overheads with encryption and metadata checks. When copying over large volumes of data you should consider only transferring files that are required, combining lots of small files into a single tar archive to reduce overheads with many separate data transfers, and compressing the data before sending it (using tools such as gzip)