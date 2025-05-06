# Retrieval-augmented Generation

FINAL GOAL: design a RAG system from scratch.

Before proceeding, start a `.py` file (or a `.ipynb` file) to work with these activities.

## Exercise 1

Suppose we are trying to search some information within the materials from this course. The course is reasonably well organized, and everything is within jupyter notebooks. Jupyter notebook files (.ipynb) are actually in JSON format.

Choose any `.ipynb` file and open it using the `json` library in Python. Investigate:

1. How can we find a cell within the notebook?
1. How can we known if the cell contains Python code or Markdown annotations?

## Exercise 2

The simplest way to search for text is using keywords. Improve your code so that it:

1. Collects a keyword from the user, and
1. Indicates all notebooks/cells that contain that keyword.

Reflect: what is the best way to present these results to the user?

## Exercise 3

There is an inherent fragility in the previous system: it requires the user to guess the keyword correctly. Trivia fact: in the early 2000s, the hability to guess keywords in Google had the same hype that using AI chatbots has nowadays.

We could prevent our user from trying to guess the exact keyword or phrase, afterall, we have an estimator for phrase similarity: BERT!

Improve your code so that it:

1. Collects a phrase from the user,
1. Calculates the phrase embedding $q$ using the CLS token from BERT
1. Traverses the course material calculating the embedding $x_i$ for each cell
1. Finds the $k$ (try with $k=1$, then generalize to any $k$) cells with minimal cosine distance ($d = \frac{ <q, x_1>}{||x|| ||c_i||}$) with relationship to the phrase.

Reflect: was this a better choice for retrieval? How can we measure this difference? (tip: research how information retrieval systems are evaluated!)

## Exercise 4

Now let's leave our retrival system waiting for a while.

Make a small program that:

1. Collects a question from the user
1. Uses an API to redirect this question to an LLM, and immediately returns the answer.
1. Add prompt information so that your answers can only regard NLP-related subjects (these are called "safeguards")

## Exercise 5

Now, let's joint everything.

We are able to find specific information from our courseware. Also, we are able to use LLMs. Use both abilities to:

1. Collect a question from the user
1. Retrieve the $K$ most relevant cells from the course material
1. Use the content of these cells as part of a prompt. The prompt includes both the question and the content from the relevant cells.
1. Phrase your prompt so that the LLM can only return information that is contained in the course material.

Reflect: how does this compare to the system in Exercise 4? How can we measure the differences?

## Exercise 6

If you have reached this far, let's start optimizing our systems.

To do so:

1. Identify which step of your processing pipeline takes the longer
1. Study if there are techniques or data structures that can make this specific step faster
1. If possible, implement the optimization and test the results.
1. Iterate until you cannot optimize anymore.

