# Machine Learning Cheatsheet

[View The Cheatsheet](http://ml-glossary.readthedocs.io/en/latest/)

## How To Contribute

1. Clone Repo
```
git clone https://github.com/bfortuner/ml-glossary
```

2. Install Dependencies
```
# Assumes you have the usual suspects installed: numpy, scipy, etc..
pip install sphinx sphinx-autobuild
pip install sphinx_rtd_theme
```

3. Preview Changes
```
cd docs
make html
```

4. Verify your changes by opening the `index.html` file in `_build/`

5. [Submit Pull Request](https://help.github.com/articles/creating-a-pull-request/)



## Style Guide

Each entry in the glossary MUST include the following at a minimum:

1. **Concise explanation** - as short as possible, but no shorter
2. **Citations** - Papers, Tutorials, etc.

Excellent entries will also include:

1. **Visuals** - diagrams, charts, animations, images
2. **Code** - python/numpy snippets, classes, or functions
3. **Equations** - Formatted with Latex

The goal of the cheatsheet is to present content in the most accessible way possible, with a heavy emphasis on visuals and interactive diagrams. That said, in the spirit of rapid prototyping, it's okay to to submit a "rough draft" without visuals or code. We expect other readers will enhance your submission over time.


## Top Contributors

We're big fans of [Distill](http://distill.pub/prize) and we like their idea of offering prizes for high-quality submissions. We don't have as much money as they do, but we'd still like to reward contributors in some way for contributing to the glossary. In that spirit, we plan to publish a running table of top authors based on number and quality of original submissions:

### Top Entries

| Entry         | Author        | Link         |
|:------------- |:------------- |:-------------|
| Example Entry | Your Name     | Your GitHub  |
| Example Entry | Your Name     | Your GitHub  |

### Most Entries

| Author        | Entries  | Link         |
|:------------- |:---------|:-------------|
| Your Name     | 24       | Your GitHub  |
| Your Name     | 18       | Your GitHub  |

We'd also like to publish top entries to our Medium Blog, for even more visibility. But in the end, this is an open-source project and we hope contributing to a repository of concise, accessible, machine learning knowledge is enough incentive on its own!


## Tips and Tricks

* [Adding equations](http://www.sphinx-doc.org/en/stable/ext/math.html)
* Quickstart with Jupyter notebook template
* Graphs and charts
* Importing images
* Linking to code


## Resources

* [How To Submit Pull Requests](https://help.github.com/articles/creating-a-pull-request/)
* [RST Cheatsheet](http://docutils.sourceforge.net/docs/user/rst/quickref.html)
* [Markdown Cheatsheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet)
* [Citation Generator](http://www.citationmachine.net)
* [MathJax Cheatsheet](https://math.meta.stackexchange.com/questions/5020/mathjax-basic-tutorial-and-quick-reference)
* [Embedding Math Equations](http://www.sphinx-doc.org/en/stable/ext/math.html)
* [Sphinx Tutorial](https://pythonhosted.org/an_example_pypi_project/sphinx.html)
* [Sphinx Docs](http://www.sphinx-doc.org/en/stable/markup/code.html)
* [Sphinx Cheatsheet](http://openalea.gforge.inria.fr/doc/openalea/doc/_build/html/source/sphinx/rest_syntax.html)

