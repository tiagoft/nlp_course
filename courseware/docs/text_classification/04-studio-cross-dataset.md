# Cross-domain sentiment analysis

## Learning outcomes

At the end of this project, students should be able to:

* Understand technical documentation and filter information to build a functioning Python program, hence developing autonomy in dealing with new libraries in Python;
* Develop a

Please, mind of these skills while working in this activity.

## History of this activity

One very big problem we have when working with labeled datasets and supervised learning is called *data drift*. This is a shift of the data content when we collect it again in a different time, place, or situation. For example, we could get some data to recognize young people slangs, but if we trained our data with a collection from the 1970s, the system would probably not work well if the production dataset contains data from the 2020s.

In this activity, we will estimate some data drift in sentiment analysis.

We have seen that, so far, our work in sentiment analysis used the IMDB dataset, and the results were reasonable. However, movie reviews are not the only situations in which we need sentiment analysis. We can also find this useful to analyze customer reviews for services and products, and to identify the general gist of what people are sharing in social media.

In this project, we will build a **cross-dataset experiment on sentiment analysis**

## Instructions

1. Locate at least 2 datasets, labeled for sentiment analysis, other than the IMDB sentiment analysis dataset (hints: https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment or https://huggingface.co/datasets/fancyzhx/yelp_polarity are good choices!).
1. Perform a same-dataset train-test experiment for each of the datasets. Use the classifier we developed before. Check the accuracy.
1. Now, perform a cross-dataset train-test: train your classifier in one dataset, and test in the others. Check the accuracy.
1. Make a figure presenting your results. Submit the figure on Blackboard as a PDF. The figure should be self-sufficient to explain what has been done. If necessary, use a figure caption to briefly explain the experiment.

## More challenges

For each of the following challenges, add the result figure as PDF to your submission.

1. (medium) We might argue that the differences in accuracy are due to the differences in the training dataset sizes. Make an experiment that controls the training set size to counter this argument.
1. (hard) We might argue that using more data will necessarily lead to better results. Make an experiment in which the classifier is trained with all datasets, simulating a situation in which we have data from different natures for training. 
1. (very hard) The previous experiment might be skewed because we used data from a domain to predict labels in that same domain, so we might not actually be evaluating data drift or the effects of shifting through different domains. Make an experiment in which the classifier is trained on all datasets, except the dataset used for testing, and repeat it for each of the datasets.

## About the use of AI

It is possible that ChatGPT can do this exercise. This is absolutely true - if not now (I am writing this in the end of May 2025!), probably in the nearby future.

Remember that the objective of this project is not to present a functioning artifact per se. Instead, the objective is to develop particular soft skills that are going to be important throughout our course. Please, do not use AI as a substitute to your own reasoning. Instead, leave AI to do only some menial tasks. A good rule of thumb is never to use AI-generated code that you are unable to critically review.