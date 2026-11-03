# How to contribute

This material accepts contributions, and yours is very welcome! You can contribute **content** (fixes, examples, demonstrations, exercises you wrote) or **features** (applets, accessibility, animations). Mathematical animations use the [Manim](https://www.manim.community/) library.

The simplest way to contribute is with a coding agent, such as [Claude Code](https://claude.com/claude-code) or [Codex](https://openai.com/codex/). The repository has an `AGENTS.md` file with the contribution rules: the agent forks the repository, applies your change, checks that it follows the course conventions, and opens the pull request for you.

## The rules, in short

* **Your text, your justification.** The agent does not write educational content for you: it formats, positions, and checks the text that *you* wrote. The pull request justification must also be yours, and specific ("aspect X was ambiguous", not "I improved the text").
* **Nothing from the future.** A contribution may only use concepts that appear before it in the course sequence (the order of this website's menu).
* **Notation.** Scalars in lowercase ($x$), constants in uppercase ($N$), vectors in bold lowercase ($\boldsymbol{x}$), and matrices in bold uppercase ($\boldsymbol{A}$).

The complete rules are in [AGENTS.md](https://github.com/tiagoft/nlp_course/blob/main/AGENTS.md).

## Animations

Mathematical **animations** are made with the [Manim](https://www.manim.community/) library. The code goes in the `manim/` folder and the GIF goes next to the page that displays it.

## Step by step with an agent

### 1. Set up your environment (only once)

You will need:

* a [GitHub](https://github.com) account;
* [git](https://git-scm.com/) and the [GitHub CLI](https://cli.github.com/) (`gh`) installed;
* [Claude Code](https://claude.com/claude-code) or [Codex](https://openai.com/codex/) installed.

Then log in to GitHub from the terminal:

    gh auth login

### 2. Fork the repository and open the agent

    gh repo fork tiagoft/nlp_course --clone
    cd nlp_course

and then start the agent inside the folder:

    claude

or

    codex

### 3. Describe your contribution

Tell the agent what you want to contribute, **where** in the material, **your text** (for content contributions), and **your justification**. For example:

    I want to contribute to the material.
    Type: content.
    Where: in text_classification/02-math_of_lr.md, right after the definition of logistic regression.
    My text: "Note that the output of the sigmoid is always between 0 and 1, so it can be
    read as the probability that the text belongs to the positive class, because ..."
    Justification: I missed seeing why we can interpret the output of logistic regression
    as a probability right at the definition.

or, for a feature:

    I want to contribute a feature.
    What it is: an applet that shows how the logistic regression output changes as I drag the weights with the mouse.
    Where: on the Math of Logistic Regression page.
    Justification: I only understood the role of the weights when I computed several cases by hand;
    an applet would make this faster to explore.

The agent may ask questions, suggest formatting or position adjustments, or point out that some concept has not been covered yet at that point in the course. Answer and adjust together with it.

### 4. Review and open the pull request

Before authorizing the pull request, read the change the agent made. When everything looks right, ask the agent to open the pull request: it will use the repository's template, with your justification and the list of course content used.

After the pull request is accepted, your name appears on the [Contributions](contributions.md) page!

## Without an agent

You can also contribute without an agent: fork the repository on GitHub, create a branch, make your changes, and open the pull request filling in the template. The same rules apply.
