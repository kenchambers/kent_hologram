# Commit and Push Changes

Run git status to see all changes, analyze the diff to understand what was modified, create a well-crafted commit message that summarizes the changes, and then push to the remote repository.

Follow these steps:

1. Run `git status` to see all modified, staged, and untracked files
2. Run `git diff` to see the actual changes in detail
3. Run `git log --oneline -5` to see recent commit message style
4. Analyze the changes and create a concise, descriptive commit message that:
   - Summarizes the main purpose of the changes
   - Follows the repository's commit message convention
   - Is clear and informative (1-2 sentences max)
5. Stage all changes with `git add .` (or selectively if appropriate)
6. Create the commit with the crafted message
7. Push to the remote repository with `git push`

If there are any conflicts, errors, or unusual situations, report them to the user before proceeding.
