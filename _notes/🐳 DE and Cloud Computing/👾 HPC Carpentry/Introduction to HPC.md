# HPC Introduction main notes

- Permissions to run programs

```bash
ls -l
```

```
-rw-rw-r--. 1 jeff jeff 12534006 Jan 16 18:50 bash-lesson.tar.gz
-rw-rw-r--. 1 jeff jeff       40 Jan 16 19:41 demo.sh
-rw-rw-r--. 1 jeff jeff 77426528 Jan 16 18:50 dmel-all-r6.19.gtf
-rw-r--r--. 1 jeff jeff   721242 Jan 25  2016 dmel_unique_protein_isoforms_fb_2016_01.tsv
drwxrwxr-x. 2 jeff jeff     4096 Jan 16 19:16 fastq
```

- First column -> Permissions: 
	- `d` -> a directory
	- `r` -> read
	- `w` -> write
	- `x` -> Execute
- Second column: Owner
- Third column: Group
- Fourth column: Size of file
- Fifth column: Time last modified
- Sixth column: Filename

- Provide Executable permissions

```bash
chmod +x demo.sh
```

#### Special permissions

- read = `4`
- write = `2`
- execute = `1`

- For each user we will assign permissions based on the sum of these permissions

### Shell variables

- Provide a variable to a `.sh` file

	-	Executable file `demo.sh`

```bash
#!/bin/bash

# Call wc -l on the first argument
wc -l $1
```

- Running `demo.sh`

```bash
./demo.sh MY_FILE.csv
```

- Save output values into variables

```bash
TEST=$(ls -l)
echo $TEST
```

### LOOPS

```bash
for VAR in first second third
do
	echo $VAR
done
```

```bash
for FILE in $(ls)
do 
	echo $FILE
done
```


# Why use a cluster?

Using a cluster often has the following advantages:

1. Speed => Many CPU cores
2. Volume => More RAM and HDD storage
3. Efficiency 
4. Cost
5. Convenience

## Bash => Bourne Again SHell

## Connecting to remote HPC system

-> Connecting to a HPC system is most often done through a tool know as **SSH** => Secure shell

## Logging into the system

-> Connect to Cirrus

```bash
ssh myUsername@login.cirrus.ac.uk
```

```{.output}
yourUsername@[yourUsername@cirrus-login0 ~]$  ~]$
```

⚠️ If you ever need to be certain which system a terminal you are using is connected to then use the following command: `$ hostname`.

# Moving around and looking at things

- How to navigate around the system.

- `whoami` -> returns your username
- `pwd`
- `ls`

### Store HPC systems

- **Network filesystem**:
	- The home directory is an example of a network filesystem
	- Files are backed up
	- Files are typically slower to access
- **Scratch**
	- *scratch* space => Typically faster to use then the home directory or network filesystem => 🚨 It is not backed up
- **Work filesystem**:
	- Typically this will have higher performance than the home directory or network files ystem and will not usually be backed up. -> Files are not automatically deleted
- **Local scratch (job only)**:
	- Some systems may offer local scratch space while executing a job. -> Very fast storage but files are deleted after finishing a job
- **Ramdisk (job only)**:
	- Storage files in RAM disk while running a job


## Unzipping files

We just unzipped a `.tar.gz` file for this example. What if we run into other file formats that we need to unzip? Just use the handy reference below:

-   `gunzip` unzips .gz files
-   `unzip` unzips .zip files
-   `unrar` unzips .rar files (not available on many systems)
-   `tar -xvf` unzips .tar.gz and .tar.bz2 files

## Visualize files

- `cat`
- `head`
- `tail`
- `less`

# Unix streams

-   `stdin` is the input to a program. In the command we just ran, `stdin` is represented by `*`, which is simply every filename in our current directory.
    
-   `stdout` contains the actual, expected output. In this case, `>` redirected `stdout` to the file `word_counts.txt`.
    
-   `stderr` typically contains error messages and other information that doesn’t quite fit into the category of “output”. If we insist on redirecting both `stdout` and `stderr` to the same file we would use `&>` instead of `>`. (We can redirect just `stderr` using `2>`.)