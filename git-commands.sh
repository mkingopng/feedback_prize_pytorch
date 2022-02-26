git pull --rebase origin main

git push origin main

git pull --rebase

git push

git reset --mixed origin/main



git commit -m "This is a new commit for what I originally planned to be amended"

git push origin main

git reset --mixed origin/main

# create a new local repository
git clone /path/to/repository

# Create a working copy of a local repository:
git clone /path/to/repository

# add one or more files to staging index
git add <filename>
git add *

# commit changes to head (but not yet to the remote repository):
git commit -m "Commit message"

# Commit any files you've added with git add, and also commit any files you've changed since then:
git commit -a

# Send changes to the master branch of your remote repository:
git push origin master

# List the files you've changed and those you still need to add or commit:
git remote add origin <server>

# List all currently configured remote repositories:
git remote -v

# Create a new branch and switch to it:
git checkout -b <branchname>

# Switch from one branch to another:
git checkout <branchname>

# List all the branches in your repo, and also tell you what branch you're currently in:
git branch

# Delete the feature branch:
git branch -d <branchname>

# Push the branch to your remote repository, so others can use it:
git push origin <branchname>

# Push all branches to your remote repository:
git push --all origin

# Delete a branch on your remote repository:
git push origin :<branchname>

# Fetch and merge changes on the remote server to your working directory:
git pull

# To merge a different branch into your active branch:
git merge <branchname>

# View all the merge conflicts:
git diff

# View the conflicts against the base file:
git diff --base <filename>

# Preview changes, before merging:
git diff <sourcebranch> <targetbranch>

# After you have manually resolved any conflicts, you mark the changed file:
git add <filename>

# You can use tagging to mark a significant changeset, such as a release:
git tag 1.0.0 <commitID>

# CommitId is the leading characters of the changeset ID, up to 10, but must be unique. Get the ID using:
git log

# Push all tags to remote repository:
git push --tags origin

# If you mess up, you can replace the changes in your working tree with the last content in head. Changes already added to the index, as well as new files, will be kept.
git checkout -- <filename>

# Instead, to drop all your local changes and commits, fetch the latest history from the server and point your local master branch at it, do this:
git fetch origin

git reset --hard origin/master

# Search the working directory for foo():
git grep "foo()"

# git-lfs filter-process: 1: git-lfs: not found the remote end hung up unexpectedly
