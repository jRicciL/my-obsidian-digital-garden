---
---

# Docker for Application Packaging

To containerize an. application using Docker, 3 main components are distinguished:

-   Dockerfiles
-   Docker images
-   Docker registries

## Dockerfile

-   A set of instructions used to create a docker image.
-   Each instruction is an operation used to package the application.
    -   Install dependecies
    -   Compile the code
    -   Impersonate a specific user
-   building a Docker image using a Dockerfile is a lightweight and quick process.

```docker
FROM - to set the base image
RUN - to ececute a command
COPY and ADD - to copy files from the host container
CMD - to set the default command to execute when the container stars
EXPOSE - to expose an application port
```

## Docker image

-   Is a read-only template that enables the creation of a runnable instance of an application.
-   Provides the execution environment for an application
    -   Essential code
    -   Configuration files
    -   Dependencies

### Building an image

A docker image can be built from an existing Dockerfile using `docker build`

```docker
# build an image
# OPTIONS - optiona, define extra configuration
# PATH - required

docker build [OPTIONS] PATH

# Where options can be
-t, --tag - set the name and tag of the image
-f, --file - set the name of the Dockerfile
--build-arg - set build-time variables
```

Beforre distributing the Docker image to a wider audience, it is paramount to test it locally and verify if it meets the expected behavior.

### Running an image

```docker
# execute an image
# OPTIONS - optional;  define extra configuration
# IMAGE -  required; provides the name of the image to be executed
# COMMAND and ARGS - optional; instruct the container to run specific commands when it starts 
docker run [OPTIONS] IMAGE [COMMAND] [ARG...]

# Where OPTIONS can be:
-d, --detach - run in the background 
-p, --publish - expose container port to host
-it - start an interactive shell
```

The following example shows how to run a python app from a created image.

```docker
# run the `python-helloworld` image, in detached mode and expose it on port `5111`
docker run -d -p 5111:5000 python-helloworld
```

-   In this case `5111` is the host port that we use to access the application.
-   The `5000` is the container port that the application is listening to for incoming requests.

### Docker log

To retrieve the Docker container logs use the `docker log {{CONTAINER_ID}}`

## Docker Registry

-   It is the last step in packaging an application
    
-   Image needs to be pushed to a **public Docker image registy**
    
    -   DockerHub
    -   Habor
    -   Google Container Registy
-   Before is recommended o tag it first.
    
    -   if a tag is not provided (via the `-t` or `--tag` flag), then the image would be allocated an ID
-   To tag an existing image use the `docker tag` command:
    
    ```docker
    # tag an image
    # SOURCE_IMAGE[:TAG]  - required and the tag is optional; define the name of an image on the current machine 
    # TARGET_IMAGE[:TAG] -  required and the tag is optional; define the repository, name, and version of an image
    docker tag SOURCE_IMAGE[:TAG] TARGET_IMAGE[:TAG]
    ```
    
-   Once the image is tagged → push the image to a registry: `docker push`
    
    ```docker
    # push an image to a registry 
    # NAME[:TAG] - required and the tag is optional; name, set the image name to be pushed to the registry
    docker push NAME[:TAG]
    ```
    

## Useful docker comands

### Build images

```bash
docker build [options] path
```

### Run images

```bash
docker run [options] image [command] [arg...]
```

### Get logs

```bash
docker loags CONTAINER_ID
```

### List images

```bash
docker images
```

### List containers

```bash
docker ps
```

### Tag images

```bash
docker tag SOURCE_IMAGE[:TAG] TARGET_IMAGE[:TAG]
```

### Push images

```bash
docker push NAME[TAG:]
```