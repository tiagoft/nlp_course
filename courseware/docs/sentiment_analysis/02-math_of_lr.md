# The Mathematics of Logistic Regression

## Learning outcomes

At the end of this activity, students should be able to:

* Describe the vectorizer + classifier pipeline
* Fit and evaluate a classifier;
* Investigate scikit learn's documentation
* Relate the logistic regression mathematical model to its implementation

Please, mind of these skills while working in this activity.

## Introduction

If everything went well, at this point you are well acquainted with the problem of [classifying a text based on its words](primer.md). Let's take a dive into that problem.

ANEW data provided us with a list of words and their corresponding *valence*. Valence is the property of a sentiment that makes it "pleasant" or "unpleasant" - an this is why this is called the "pleasure" axis. We usually associate positive sentiments with a high valence, or a high pleasure. So, let's say we have the words *happy*, *sad*, *joy*, and *anger*. Their valences (or: their values in the pleasure axis) are somewhat like:

| Word | Valence |
| --- | --- |
| happy | high |
| sad | low |
| joy | high |
| anger | low |

In our exercise, it is likely that we had many different solutions to use this information to classify our texts. However, we will now focus on a very specific one.

## Summing values

Let's attribute values to our words. If the word has high valence, it receives a value of $1$. If the word has low valence, it receives a value of $-1$. If it is not on our list, it receives the value of $0$. Then, we can calculate the sum of the values for all words for the whole phrase. For example, in the phrase: "I feel angry", we have values $0$, $0$, and $-1$. Hence, the final sum is $-1$.

**Exercise**: calculate the sum of all word values in the phrases below:

<details>
<summary>I am so happy!</summary>
1
<details><summary>Why?</summary>
0 (I) + 0 (am) + 0 (so) + 1 (happy)
</details>
</details>

<details>
<summary>I am so sad!</summary>
-1
</details>

<details>
<summary>I am so sad, but the world if full of joy!</summary>
0
</details>

## Weighted averages

Now, let's formulate this in a more elegant way. We will divide our sum into two vectors. Each of these two vectors is as long as our vocabulary. In our case, we have four words in our vocabulary, hence our vectors should have $4$ elements.

### A bag-of-words

The first vector will be called $x$. It contains one position (or: one dimension) for each word in the vocabulary. Each of the elements is called $x_i$, where $i$ is the position, that is:

$$ 
x = 
\begin{bmatrix} 
x_1 & x_2 & x_3 & x_4 
\end{bmatrix}
$$

$x$ will represent the presence of each word in our text, that is, $x_i=1$ if the corresponding word is present in the text, and $x_i=0$ otherwise. Of course, we need to have some way to link words with their respective $i$'s. One way to do that is to use a table:

| Word | Value of $i$ |
| --- | --- | 
| happy | 1 | 
| sad | 2 | 
| joy | 3 | 
| anger | 4 |

This means that each text will have a different $x$. Find $x$ for each of the following phrases:

<details>
<summary>I am so happy!</summary>
\begin{bmatrix} 1 & 0 & 0 & 0 \end{bmatrix}
<details>
<summary>Why?</summary>
It has the word happy, but not the words sad, joy, or anger.
</details>
</details>

<details>
<summary>I am so sad!</summary>
\begin{bmatrix} 0 & 1 & 0 & 0 \end{bmatrix}
</details>

<details>
<summary>I am so sad, but I am somewhat happy because the world if full of joy!</summary>
\begin{bmatrix} 1 & 1 & 1 & 0 \end{bmatrix}
</details>


This type of representation is usually called **Bag-of-Words**. This is because the order of words is ignored, as if the words were put into a bag.

### The weight vector

Now, we will define a weight vector $w$. It has the same number of elements of $x$ (that is, in our case, it is $4$). Each element $w_i$ contains the weight given to that word, following the rule we had used previously ($1$ for positive valence, $-1$ for negative valence, $0$ for neither positive or negative valence).

**Exercise**

Find the values for:
$$w =\begin{bmatrix}w_1 &w_2 &w_3 &w4\end{bmatrix} $$

in our current problem.

<details>
<summary>Answer here</summary>
w=\begin{bmatrix} 1 & -1 & 1 & -1 \end{bmatrix}
</details>

## Multiplying vectors

Now, to replicate the same result we had in the [Summing Values](#summing-values) section, we need to use the weights in $w$ and multiply them by the values in $x$. Also, we can add a bias $b$ to adjust the "zero" in our model. In other words, we need to calculate:

$$
z = b + \sum_{i=1}^n x_i w_i = b+ x_1w_1 + x_2w_2 + x_3w_3 + x_4w_4
$$

Note that this is equal to using a matrix multiplication:

$$
z = b + \begin{bmatrix} x_1 & x_2 & x_3 & x_4 \end{bmatrix} \begin{bmatrix} w_1 \\ w_2 \\ w_3 \\ w_4 \end{bmatrix}
$$

And, for this reason, the literature usually uses the compact matrix notation:

$$
z = x w^T + b
$$

The matrix notation is often called "elegant" because it is compact and clear. However, our problem does not end here.

The $x$ vector represents the document. The $w$ vector represents the weights given to each of these words. So we can start with the question: **what are the optimal values for $x$ and $w$?**

## Types of Bag-of-Words representations


The simplest way to determine $x$ is to use a binary representation where $x_i$ is assigned a value of 1 if the corresponding word exists in the text and 0 if it does not. An alternative method is to define $x_i$ as the number of times the corresponding word appears in the text. This value is known as **Term Frequency (TF)** and provides a measure of how frequently a word appears within a given document. A higher term frequency suggests greater relevance within that specific text.

A more refined approach is to adjust the term frequency by incorporating document frequency. Specifically, TF is divided by the number of documents in which the word appears, referred to as **Document Frequency (DF)**. This results in the Term Frequency–Inverse Document Frequency (TF-IDF) method, which produces a vector indicating the significance of each word in distinguishing one document from the entire collection. By reducing the weight of commonly used words across multiple documents, TF-IDF enhances the relevance of terms that are more unique to a given text.

**Exercises**

<details><summary>What does Term Frequency (TF) measure?</summary>TF quantifies how often a word appears in a document. A higher frequency suggests greater relevance within that specific document.</details>
<details><summary>What is the purpose of Inverse Document Frequency (IDF) in the TF-IDF method?</summary>IDF adjusts the TF score by reducing the importance of words that appear in many documents. This helps highlight terms that are more unique to a particular document.</details>
<details><summary>If a word appears in 100 documents out of a total of 1,000 appearances in unique documents, how does IDF affect its relevance?</summary>The IDF component lowers the word's significance because it is relatively common across documents. Words appearing in fewer documents receive higher importance in TF-IDF.</details>

## Logistic Regression

You may have noted that the equation $z=b+\sum_{i=1}^n x_i w_i=b+x w^T$ is equivalent to the formula for a linear prediction. This means that $z$ can assume any real number, positive or negative. However, we can ally a function called *logistic function* to $z$ so that it becomes bounded to the limits $[0,1]$:

$$
y=\sigma(z)=\frac{1}{1+e^{-z}}.
$$

The logistic function $\sigma(.)$ has interesting properties:

* If $z \rightarrow \infty$, $\sigma(z)=1$ (or: $\lim _{z\rightarrow \infty}=1$)
* If $z \rightarrow -\infty$, $\sigma(z)=0$ (or: $\lim _{z\rightarrow -\infty}=0$)
* If $z=0$, $\sigma(z)=\frac{1}{2}$.

Because of that, the result $y$ can be interpreted as a probability $P(\text{class is ``positive''} | x)$, that is, the probability of the input belonging to the "positive" class given its representation $x$. This is commonly written as $P(C=c_j | X=x)$, meaning that there are two random variables $C$ and $X$ and that they relate to each other by means of the logistic regression $P(C|X) = \sigma(xw^T+b)$.

**Exercises**

<details><summary>What does the random variable X represent?</summary>X represents the distribution of all vectors that represent texts within a collection. A sample drawn from X is the representation of a text.</details>

<details><summary>What does the random variable C represent? What is P(C) and P(C|X)?</summary>C represents a variable whose samples are classes. P(C) is the probability distribution of classes, that is, the probability of drawing a random sample from the dataset and it belonging to each of the classes. P(C|X) represents the probability of C given X, that is, the probability of drawing a random sample and it belonging to each of the classes, but with prior knowledge of its representation. Ideally, P(C|X) is 1 for the correct class and 0 for all other classes.</details>

## Finding the weights and biases

Although we could adjust all coefficients for our logistic regression by hand, a more effective way is to find the difference between the predicted $P(C|X)$ and a ground truth distribution derived from a manually labeled dataset. There are many methods for this, and the most common one is [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent). In classification problems, gradient descent is used to minimize the [cross entropy](https://en.wikipedia.org/wiki/Cross-entropy) between the predicted and the ground-truth distributions.

### Creating a classification pipeline

At this point, we are not going to reimplement gradient descent. Instead, we are going to use the scikit-learn library for that. Note that we need to create and object perform TFIDF vectorization (as in: convert a text to a vector using TFIDF), another object to perform Logistic Regression per se, and, finally, join them using a Pipeline object:

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    model = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression())
    ])

### Dividing data into train and test

When we use our model, we need to fit it to a portion of our dataset, and then use the remainder of our labeled dataset to evaluate how our model would behave on unseen data. For such purpose, we use the `train_test_split` function:

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

When dealing with texts, `X` is assumed to have one text per line. In fact, `X` can be a list of texts, a `pandas.Series` containing one text per line, or even a `numpy.array` with strings. `y` is an equivalent structure containing one label per line.

### Fit and evaluate our model

Last, we need to fit (adjust weights and biases) our model to our training data, and then evaluate it in the testing data. We can evaluate the model using the accuracy score, which is the number of correctly classified dataset items divided by the total number of items in the dataset:

    from sklearn.metrics import accuracy_score
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')

The `.fit` method finds both the vocabulary and the parameters for the TFIDF vectorizer and the logistic regression. The `.predict` method generates predicted labels for each element in `X_test`.

**Exercise**:

Consider the diagram below, which depicts a pipeline similar to that explained in this section. In this figure, $A$ and $B$ are vectors.

<div class="mermaid">
graph LR;
    T([Text]) --> V[Bow Vectorizer] --> A([A]) --> Classifier --> B([B]);
</div>

<details><summary>What is the dimension of A?</summary>A is the result of a bag-of-words vectorizer. It is a vector with as many positions (dimensions) as there are words in the vocabulary.</details>

<details><summary>What is the dimension of B?</summary>B is the result of a logistic regression. If it follows the formulation shown in this lesson, it is one single real number between 0 and 1.</details>


# Practical exercise

1. Use the same dataset you gathered for [Activity 1](primer.md).
2. Split it into train and test.
3. Compare the performance of the rule-based classifier you had done previously with the performance of a logistic regression trained on data.
4. Use [scikit learn's documentation](https://scikit-learn.org/stable/). Find the *vocabulary* of your trained model. How many words does it have?
5. Still using the documentation, find the variables that contain the weights $w$ and the bias $b$ within your model.
6. Relate the $z$ and $y$ variables in this lesson to the methods `.predict_proba()` and `.decision_function()`. Which is which?
