---
---

# Container LifeTime

## Section overview
- Defining the problem of persistent data
- Key concepts with containers: immutable, ephemeral
- Learning and using Data Volumes
- Learning and using Bind Mounts

## Container LifeTime
- Containers are usually immutable and ephemeral
- *immutable infrastructure*: only re-deploy containers, never change
- Two solutions for persistent data:
	- ==Volumes==:
		- Make special location outside of container UFS
	- ==Bind Mounts==:
		- Link container path to host path

## Data Volumes
- `VOLUME` command in a `Dockerfile`
- Volumes require manual deletion:
	- Even if the container is removed, the volume will continue
- To list all the available `Volumnes`
```bash
docker volume ls
```
- <mark style='background-color: #FFA793 !important'>By default, </mark> volumes have no name. 
- To name a volume when running a container

```bash
docker container run -d -v my-volume:/var/lib/volume
```

## Bind Mounting

- Maps a host file or directory to a container file or directory
	- `host dir` <-- --> `container dir`
- Basically just two locations pointing to the same *physical* location
- Can't use in a `Dockerfile`, must be a `container run `
	- They should be used at runtime
	
#### Path mapping

```bash
# The full path
... run -v /path/to/the/HOST_dir:/path/container
```

#### Path mapping example

- Create a container with a mapped directory
```bash
docker container run -d --name nginx2 -p 4000:80 -v $(pwd):/usr/share/nginx/html
```

- Edit and add files from the `host` directory
- Edit files from the `container` directory
	- Run the container's terminal using `bash`
```bash
# Run the container
docker container exec -it nginx2 bash

# Go to the shared directory
cd /usr/share/nginx/html/

# Create a file from the container
touch created_from_container.txt
echo 'This was created from the container command line' > created_from_container.txt
```


## Quiz on Persistent data
- Which type of persistent data allows you to attach an ==existing directory== on your host to a directoy inside of a container?
	- **Bind Volume**

- When adding a bind mount to a docker run command, you can use the shortcut `$(pwd)`. What does that do?
	- Runs the shell command to print the current working directory to avoid having type out the entire of the directory path

- When making a new volume for a mysqul container, where could you look to see where the data path should be located in the container?
	- `Docker hub`: looking through the `README.md` or `Dockerfile` of the mysql official image, you could find the database path documented or the `VOLUME` stanza

## Assignment: Database Upgrades with Named ==Volumes==

- Database upgrade with containers.
1. Create a `postgres` container with named volume `psql-data` using version `9.6.1`
2. Use *Docker Hub* to learn `VOLUME` path and versions needed to run it
3. Check logs, stop container
4. Create a new `postgres` container with the same named volume using `9.6.2`
	1. Both containers will share the same volumn

### Solution
- In Docker Hub we found that the stored data of the container is put in the following directory:
	- `/var/lib/postgresql/data`
- Let's create the first container with a ==volume== named `psql`

```bash
# Link the `psql` VOLUME to the path
docker container run -d --name psql1 -v psql_vol:/var/lib/postgresql/data postgres:9.6.1
```

- Check the logs
```bash
docker container logs -f psql
```

- Stop this container
```bash
docker container stop psql
```

- Create the second container with a `9.6.2` version and linked to the same path
```bash
# Link the `psql` VOLUME to the path
docker container run -d --name psql2 -v psql_vol:/var/lib/postgresql/data postgres:9.6.2
```