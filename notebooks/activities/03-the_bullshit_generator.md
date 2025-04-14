# The Bullshit Generator

Bullshit is the name given to phrases that make absolutely no sense, but are somewhat well formed.

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
