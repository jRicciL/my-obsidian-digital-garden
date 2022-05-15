# Working on a remote HPC system



<div class="rich-link-card-container"><a class="rich-link-card" href="https://carpentries-incubator.github.io/hpc-intro/12-cluster/index.html" target="_blank">
	<div class="rich-link-image-container">
		<div class="rich-link-image" style="background-image: url('https://carpentries-incubator.github.io/hpc-intro/assets/favicons/incubator/favicon-196x196.png')">
	</div>
	</div>
	<div class="rich-link-card-text">
		<h1 class="rich-link-card-title">Working on a remote HPC system – Introduction to High-Performance Computing</h1>
		<p class="rich-link-card-description">
		
		</p>
		<p class="rich-link-href">
		https://carpentries-incubator.github.io/hpc-intro/12-cluster/index.html
		</p>
	</div>
</a></div>



### Secure connections

- Establish a connection to the cluster
- `ssh userName@hostname` => 


Generate a new public-private key pair using the following command, which will produce a stronger key than the `ssh-keygen` default by invoking these flags:

-   `-a` (default is 16): number of rounds of passphrase derivation; increase to slow down brute force attacks.
-   `-t` (default is [rsa](https://en.wikipedia.org/wiki/RSA_(cryptosystem))): specify the “type” or cryptographic algorithm. `ed25519` specifies [EdDSA](https://en.wikipedia.org/wiki/EdDSA) with a 256-bit key; it is faster than RSA with a comparable strength.
-   `-f` (default is /home/user/.ssh/id_algorithm): filename to store your private key. The public key will be identical, with a `.pub` extension added.

```
[user@laptop ~]$ ssh-keygen -a 100 -f ~/.ssh/id_ed25519 -t ed25519
```


If `~/.ssh/id_rsa` already exists, you will need to specify choose a different name for the new key-pair. Generate it as above, with the following extra flags:

-   `-b` sets the number of bits in the key. The default is 2048. EdDSA uses a fixed key length, so this flag would have no effect.
-   `-o` (no default): use the OpenSSH key format, rather than PEM.

```
[user@laptop ~]$ ssh-keygen -a 100 -b 4096 -f ~/.ssh/id_rsa -o -t rsa
```

When prompted, enter a strong password that you will remember. There are two common approaches to this:

1.  Create a memorable passphrase with some punctuation and number-for-letter substitutions, 32 characters or longer. Street addresses work well; just be careful of social engineering or public records attacks.
2.  Use a password manager and its built-in password generator with all character classes, 25 characters or longer. KeePass and BitWarden are two good options.

### SSH agent

#### SSH Agents on Linux, macOS, and Windows[](https://carpentries-incubator.github.io/hpc-intro/12-cluster/index.html#ssh-agents-on-linux-macos-and-windows)

Open your terminal application and check if an agent is running:

```
[user@laptop ~]$ ssh-add -l
```

-   If you get an error like this one,
    
    ```
    Error connecting to agent: No such file or directory
    ```
    
    … then you need to launch the agent _as a background process_.
    
    ```
    [user@laptop ~]$ eval $(ssh-agent)
    ```
    
-   Otherwise, your agent is already running: don’t mess with it.
    

Add your key to the agent, with session expiration after 8 hours:

```
[user@laptop ~]$ ssh-add -t 8h ~/.ssh/id_ed25519
```

```
Enter passphrase for .ssh/id_ed25519: 
Identity added: .ssh/id_ed25519
Lifetime set to 86400 seconds
```

For the duration (8 hours), whenever you use that key, the SSH Agent will provide the key on your behalf without you having to type a single keystroke.

### Nodes

![[Pasted image 20220305201505.png]]

- A node includes the same components that a laptop:
	- CPUs => Cores or processors
	- Memmory => RAM
	- Disk space

#### loging nodes
- individual computers that compose a cluster are typically called nodes
	- The initial node is clled the *head node*, *loging node*, *landing pad*, or *submit node*
	- A loging node serves as an access point to the cluster
	- This node should not be used for time-consuming or resource-intensive task

#### Dedicated Transfer nodes
- There are nodes dedicated for data transfer only.
- As a rule of thumb, consider all transfers of a volume larger than 500 MB to 1 GB as large.

#### Worker nodes

- `sinfo` => View all worker nodes

### Explore the head node

```
[user@laptop ~]$ ssh yourUsername@graham.computecanada.ca
[yourUsername@gra-login1 ~]$ nproc --all
[yourUsername@gra-login1 ~]$ free -m
```

You can get more information about the processors using `lscpu`, and a lot of detail about the memory by reading the file `/proc/meminfo`:

```
[yourUsername@gra-login1 ~]$ less /proc/meminfo
```

You can also explore the available filesystems using `df` to show **d**isk **f**ree space. The `-h` flag renders the sizes in a human-friendly format, i.e., GB instead of B. The **t**ype flag `-T` shows what kind of filesystem each resource is.

```
[yourUsername@gra-login1 ~]$ df -Th
```

### Explore the worker nodes

Let’s look at the resources available on the worker nodes where your jobs will actually run. Try running this command to see the name, CPUs and memory available on the worker nodes:

```
[yourUsername@gra-login1 ~]$ sinfo -n aci-377 -o "%n %c %m"
```