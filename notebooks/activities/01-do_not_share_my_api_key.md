# Don't share your API key

Security in development projects is a real issue, because sometimes we commit sensitive data to our repository - an all of a sudden, our API keys, e-mails, and so on, can appear online for everyone to see.

Luckily, this type of information can be easily detected using regex.

In this activity, you will create a Git hook. A Git hooks are scripts, placed under the `.git/hooks/` directory, that is executed under particular conditions. The filename defines when the hook will be executed - when we create a git repository, we get many templates for hooks within the directory.

We are especially interested in the pre-commit hook. What we want to do is: *before* the commit is accepted, we swipe all files finding sensitive information and, if any sensitive information is found, the commit is aborted. To abort a commit, use `exit(1)` at any point of the script.

Sensitive information could be your e-mail, your document numbers, or, even worse, some API KEY from cloud services - check, for example, OpenAI's or Google Gemini's typical api key.

You might want to start from a functional python script for the pre-commit hook. Use the code below as a template:

## Example hook in Python

    import os
    import subprocess

    def get_staged_files():
        """Retrieve the list of staged files."""
        result = subprocess.run(['git', 'diff', '--cached', '--name-only'], capture_output=True, text=True)
        return result.stdout.splitlines()

    def count_newlines(file_path):
        """Count the number of newline characters in the given file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return sum(1 for _ in f)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return 0

    if __name__ == "__main__":
        print("Checking newline characters in staged files...")

        for file in get_staged_files():
            if os.path.isfile(file):
                newline_count = count_newlines(file)
                print(f"File: {file} | New lines: {newline_count}")

        # Exit successfully to allow commit to proceed
        exit(0)

## Keep going

If you want to keep going, try this experiment: crawl through github repositories, and use your functions to seek for sensitive information. If you find any, let the authors know (privately), so that they can solve the issue.
