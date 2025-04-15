# How much data do we need? A study on detecting Fake News with Machine Learning

One amazing feat we have today is that there are so many datasets available online. One important hub for datasets is Kaggle.

One important problem we've had in the last few years is the propagation of fake news - which are statements that range from being outright lies, to convenient interpretations of the facts, to naive replications of common sense.

Well, fortunately, we have many datasets on Kaggle containing news labeled as either fake or true. But: is it feasible to use Machine Learning to detect fake news?

## Single dataset results

Start by downloading a fake news dataset. Run the usual train-test pipeline and check the accuracy you could classify fake news from non-fake news. Is this accuracy enough to use your system as a filter for fake news in a social media portal?

## Cross-dataset study

Now, download *another* two fake news datasets. First, run the train-test pipeline evaluation in each one of them. After that, proceed to a more interesting experiment:

* *train* a classification pipeline in one dataset
* *test* the pipeline in *another* dataset

Do this for all dataset combinations you have. What happens to the results?

Do you believe fake news detection with a classifier like ours is reliable?

## Keep going

Find how much each word contributes to the overall classification. Are these words the same in each dataset?

Reflect: 

* Why is this happening? 
* Do these results indicate that our detector is reliable?
* Is it feasible to detect fake news using a Bag-of-Words classifier?
