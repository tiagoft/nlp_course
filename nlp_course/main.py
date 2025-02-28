"""
My course on NLP - A course on modern NLP
Made from cookiecutter template:
cookiecutter gh:tiagoft/hello_world
"""

import importlib.resources
from rich.console import Console
import typer

import nlp_course as nlp

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command('crawl')
def crawl_url(
    url: str,
    max_documents: int = 10,
):
    """
    Crawl a website and print the visited URLs
    """
    _, saved_info, saved_emails = nlp.crawl(url, max_documents)
    console.print(saved_info)
    console.print(saved_emails)


@app.command('info')
def print_info(custom_message: str = ""):
    """
    Print information about the module
    """
    console.print("Hello! I am My course on NLP")
    console.print(f"Author: { nlp.__author__}")
    console.print(f"Version: { nlp.__version__}")
    if custom_message != "":
        console.print(f"Custom message: {custom_message}")


@app.command('demo')  # Defines a default action
def run():
    """
    This is a demonstration function that shows how to use the module
    """
    print("Hello world!")
    nlp.my_function()
    asset_file = importlib.resources.open_text('nlp.assets', 'poetry.txt')
    print(asset_file.read())
    asset_file = importlib.resources.open_text('nlp.assets.test_folder',
                                               'test_something.txt')
    print(asset_file.read())


if __name__ == "__main__":
    app()
