# Agents

The idea of agents started in cybernetics. The underlying idea was that it would be possible to develop small-scale devices, and their interaction would lead to emergent behaviors. One example of such is the Neural Network: we model a single neuron, and the interaction of many neurons creates an emergent behavior: the ability to perform regression.

In the context of LLMs, agents are objects that solve an individual task: we create an agent that parses information from text, another one that processes parsed information, and so on. Usually, using agents lead to simpler prompts, which can reduce the chance of making mistakes.

In this work, we will assume an agent is simply a character in a narrative. It has two components: a role, which defines its behavior and how it reacts to external inputs, and a memory, which 

    class MyAgent:
        def __init__(self, role : str):
            self.role = role
            self.memory = ""

        def listen(self, new_info : str):
            pass

        def talk(self):
            pass

## Exercise 1

Create an agent which is a character in a story. It does not have to be useful - you might want to create an entirely fictional character. For example:

You are Private Detective John Jackson. You live in New York, 1929. The great depression has just arrived, and, with that, business has decreased. You are worried about how you are going to pay the next rent. Under the dim light of your office, you read the news.

Or, you might want to create a serious agent:

You are a research assistant. You read papers and summarize them. You will receive a text containing a scientific paper and will summarize it, highlighting its main claims and the evidence supporting each claim. Also, you will find out why each claim is relevant to the paper's target audience.

After that, implement the `listen` method. This method adds some new info to the agent's memory.

Finally, implement the `talk` method. This method sends the role and the memory to the LLM API and retrieves a response from the agent. Optionally, you might want to store these results in the memory (like an agent that can listen to itself).

Test your agent.

## Exercise 2

Now, create an agent that interacts with your first agent. It has the same structure, but has a different role. You might want to borrow another character from our noir movie:

You are Mrs. Jannette Kluperskien Lappone. Your father immigrated from Austria to New York when you were 15, and you promptly married a rich entrepreneur from Boston called William Lappone. William had kindly supported your family, but with the financial crisis he lost everything and was found dead in his office. The police wants to rule this as suicide, which would prevent you from getting your late husband's life insurance. You want to hire a private detective to find out who murdered your husband.

Or, you might want someone that interacts with your primary content generator:

You are an undergraduate student. You want to understand everything a research assistant writes, but you struggle with logic and with difficult words. When given a research summary, you ask clarification questions.

Now, make your agents interact: one of them talks, and then the other one listens, and so on. You might want to quickstart the conversation giving both agents some prompt to start with.

What happens to narrative dialogue?

What happens to the quality of a summary after some interactions?

In special: do you see convergence, or do you see a progression?


