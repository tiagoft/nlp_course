# Sentiment Analysis with ANEW

## Learning outcomes

At the end of this project, students should be able to:

* Understand technical documentation and filter information to build a functioning Python program, hence developing autonomy in dealing with new libraries in Python;
* Read through scientific papers, understand their rationale and link that with a corresponding functionality, hence developing autonomy in reading and understanding scientific language and its links to practical problems;
* Show results in a clear, objective manner, hence developing communication skills.

Please, mind of these skills while working in this activity.

## History of this activity

Natural Language Processing (NLP) concerns the use of computers to automate tasks related to natural language. Natural language is usually in the form of writing, and we have plenty of this nowadays: text messages, blogs, social media, news, and so on.

In fact, nowadays, anyone can suddenly become an author.

This has a consequence for companies. Until before the boom of social media, the only ones who could spread information were broadcast companies (TV or radio), newspaper companies (who owned large printers!), and maybe, to some extent, music recording companies. Hence, if a company wanted to understand the public perception of their products, they only had to monitor a handful of information outlets.

Not nowadays.

Nowadays, anyone can suddenly become an author.

Anyone can suddenly become an influencer.

Hence, companies now need to monitor... everyone!

And where is everyone?

In social media, of course!

In this project, we will build a small **social media monitor** able to find the general sentiment around something.

## System overview

The system we will build works like this:

1. Choose any entity you would like (for example: "Brazil", or "Vila Olímpia", or "Banana")
1. Download posts about that entity from a social media of your choice
1. Find out if these posts are, on average, good ("positive" sentiment) or bad ("negative" sentiment)
1. Present your results

Each one of these steps has their own challenges, as described below. These challenges are expected to exist, and students are expected to deal with them autonomously as possible. In fact, there is a multitude of solutions for each one of them.

### Choice of an entity

You may choose any entity you are interested in. This project works better if the entity is well-known. For example: you might want to find out what people are talking about *you* on social media, but chances are that there are going to be very few posts specifically about you. If you have no idea where to begin, start with "Brazil".

*Expected challenges:* try not to choose controversial entities here. Other than that, this step should not take you more than a few minutes.

### Downloading posts

Now, you will need to choose a social media outlet. At this point, you are expected to understand how to make a computer program that downloads information from the Internet and processes this information. If you haven't done so yet, check, for example, the `requests` and the `beautifulsoup` libraries in Python (these are suggestions - you are free to use any library you want; however, if you choose not to use Python, you might be putting yourself in trouble for the rest of the course).

*Expected challenges:*

* If you never used Python to perform web requests and deal with responses, take some time to experiment with the `requests` API. Build, for example, a system that downloads data from URL.
* The social media outlet you chose could have a public API. If this is the case, you are going to access an API and deal with (probably) `json` responses. Read the API documentation to understand how to access it correctly.
* Maybe the social media outlet does not have a public API. If this is the case, you might need to do some webscrapping in its contents. For such, libraries like `selenium` can be helpful.
* In any case, you are expected to read through the specific documentations and build the system yourself.

### Find the average sentiment of these posts

Now comes some NLP theory. It is generally accepted (this idea appeared in the middle of the XX Century) that the words used in a text work like units of meaning. This means that choosing, for example, words with a "negative" sentiment makes your text lean towards a more negative sentiment, using footbal metaphors makes your text read more like football, and using overly complicated words might make your text sound pedantic and boring.

One of the methods to find the sentiment of a word is to ask many people and then compute the average sentiment. This actually gave origin to the Affective norms for English words (ANEW) dictionary. This dictionary relates each 

Bradley, M. M., & Lang, P. J. (1999). Affective norms for English words (ANEW): Instruction manual and affective ratings (Vol. 30, No. 1, pp. 25-36). Technical report C-1, the center for research in psychophysiology, University of Florida. url: e-lub.net/media/anew.pdf

Open Science Foundation link (containing ANEW in CSV format): https://osf.io/y6g5b/wiki/anew/

*Expected challenges*:

* You are expected to read the ANEW article (Bradley & Lang, 1999) and understand how the ANEW dictionary was created;
* You probably don't know what Pleasure, Arousal, and Dominance are. Read the paper, and, if necessary, follow the references to understand them;
* There might be inconsistencies in ANEW. You are expected to figure out how to deal with them;
* Although we have an affective rating for each word, we do not have a rating for the whole post. You are expected to figure out a possible way to combine this information and yield a final decision on if the post is "positive" or "negative" towards your entity. If necessary, you can add other labels such as "neutral".

### Present your results

You are expected to present your results. Prepare two slides:

1. A slide briefly describing how your algorithm works. You can use figures and equations, but you should not use source code itself. The slide should clearly state the rationale of your algorithm. Assume this slide will be read by your colleagues who had different solutions for this same exercise, that is, they have a similar background, but do not know what you, specifically, have done.
1. A slie showing your results as a figure.

*Expected challenges*:

* The first slide should be brief, yet complete. You will need to be clear, but you can't simply list all possible information. It is expected that you will reflect on what is the necessary information to include on the slide. If you find useful, have some review rounds with a colleague.
* The second slide should be clear and self-sufficient. Remember you are only trying to show the sentiment towards a particular entity. What are the best types of figures (bar graphs, line graphs, etc.) that could help us convey this information? You are expected to reflect on this, and possibly experiment with more than one figure type. HINT: never use a pie chart.

## About the use of AI

It is possible that ChatGPT can do this exercise. This is absolutely true - if not now (I am writing this in the end of May 2025!), probably in the nearby future.

Remember that the objective of this project is not to present a functioning artifact per se. Instead, the objective is to develop particular soft skills that are going to be important throughout our course. Please, do not use AI as a substitute to your own reasoning. Instead, leave AI to do only some menial tasks. A good rule of thumb is never to use AI-generated code that you are unable to critically review.