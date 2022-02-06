---
---

# Version control in Data Science

## Different scenarios

### Scenario 1

- Use a new branch for a new feature
	- `master` or `main` branch
	- `develop` branch
![[Captura de Pantalla 2021-09-01 a la(s) 13.26.08.png]]

==Step 1==: Switch to `develop`branch
- `git checkout develop`
- `git pull`

==Step 2==: Create a new branch to work on a demographic feature
- `git checkout -b demographic`
- Work on the feature and commit changes:
	- `git commit -m 'added gender recommendation'`
	- `git commit -m 'Another commit with capital case'`

==Step 3==: Change to work on another feature
1. First commit the changes in the current branch `demographic`
	- `git commit -m 'refactored demographic recommendations'`
2. Change to the `develop` branch:
	- `git checkout develop`

==Step 4==: Create another branch to work on another feature
- `git checkout -b friends`

==Step 5==: Finish the work on the current branch:
- `git commit -m 'Finalized with friends'`
- `git checkout develop`

==Step 6==: Merge the `friends` into the `develop` branch.
- `git merge --no-ff friends`
- `git push origin develop`

### Scenario 2
- Working with different models evaluating their performances on the `cv` and the `train` sets:
- Check `git` history to return to a model with specific features

Let's walk through the Git commands that go along with each step in the scenario you just observed in the video.

==Step 1==: You check your commit history, seeing messages about the changes you made and how well the code performed.

- View the log history
 `git log`

==Step 2==: The model at this commit seemed to score the highest, so you decide to take a look.

- Check out a commit
`git checkout bc90f2cbc9dc4e802b46e7a153aa106dc9a88560`

After inspecting your code, you realize what modifications made it perform well, and use those for your model.

==Step 3==: Now, you're confident merging your changes back into the development branch and pushing the updated recommendation engine.

- Switch to the develop branch
`git checkout develop`

- Merge the friend_groups branch into the develop branch
`git merge --no-ff friend_groups`

- Push your changes to the remote repository
 `git push origin develop`