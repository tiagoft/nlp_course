# The Bullshit Generator

## History of this activity

Bullshit is the name given to phrases that make absolutely no sense, but are somewhat well formed. This is an introductory activity that can be executed in class (or at home?), and 

## Theory

### N-grams

By now, you might have noticed that using one single word in the past to predict the next word feels wrong. This is because we choose words based on a long-term context - and using a single word is a large oversimplification on this.

A possible solution is to change our original equation $𝑃(𝑤_𝑛∣𝑤_{𝑛−1})$ to a less naive one in which the probability of a word is calculated based on $L$ previous words ($L$ stands for "context length"): $𝑃(𝑤_𝑛∣𝑤_{𝑛−1}, w_{n-2}, \cdots, w_{n-L})$ . For such, we will need to use n-grams.

N-grams are simply sequences of N words that appear in the text. For example, in "these are nice n-grams", for n=2, we have the n-grams: "these are", "are nice", "nice n-grams". Note that now we can calculate $P(\text{nice}|\text{these are}).

### A fallback strategy

Also, by now, you probably found out that larger n-grams become more and more uncommon. This is so true that finding two texts that contain n-grams with a context $L$ larger than around 10 can be used as basis to flag copy-paste plagiarism. Hence, with larger n-grams, we will probably fall into situations in which we don't have information on how to proceed.

On the other hand, we might like larger context lengths because they can make our texts more cohesive.

How to deal with that?

One possibility is to have a weighting strategy in which the probabilities for models that consider different n-gram lengths are combined. However, the optimal combination could be hard to obtain.

Another possibility is to use a fallback strategy: we try a model with context $L$. If it fails to find the n-gram, then we proceed to a model with context $L-1$, and so on.

## Practice

In this activity, you will make a bullshit generator.

For such, the procedure is rather straightforward:

1. Download a dataset with texts that sound like what you want to generate (idea: maybe you can download many scientific articles on Natural Language Processing?)
1. Train an n-gram generator with fallback using the strategies we have seen in class
1. Go on to generate some texts.

Do they look feasible? How long did you have to stare at the texts to spot that they are bullshit?

## Keep going

Now, improve your bullshit generator. Some ideas:

1. Modify your code so that the generator cannot yield a word if that word has been yielded in the last, say, 20 generations. This is a way to make it more diverse.
1. Eventually, change words for other words. Remember to only change nouns for nouns, verbs for verbs, and so on.
