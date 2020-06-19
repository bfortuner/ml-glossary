# Machine Learning Glossary

## Looking for fellow maintainers!
Apologies for my non-responsiveness. :( I've been heads down at Cruise, buiding ML infra for self-driving cars, and haven't reviewed this repo in forever. Looks like we're getting `54k monthly active users` now and I think the repo deserves more attention. Let me know if you would be interested in joining as a maintainer with priviledges to merge PRs. 

[View The Glossary](http://ml-cheatsheet.readthedocs.io/en/latest/)

## How To Contribute

1. Clone Repo
```
git clone https://github.com/bfortuner/ml-glossary.git
```

2. Install Dependencies
```
# Assumes you have the usual suspects installed: numpy, scipy, etc..
pip install sphinx sphinx-autobuild
pip install sphinx_rtd_theme
pip install recommonmark
```
For python-3.x installed, use:
```
pip3 install sphinx sphinx-autobuild
pip3 install sphinx_rtd_theme
pip3 install recommonmark
```
3. Preview Changes

If you are using, make build.

```
cd ml-glossary
cd docs
make html
```

For Windows. 

```
cd ml-glossary
cd docs
build.bat html
```


4. Verify your changes by opening the `index.html` file in `_build/`

5. [Submit Pull Request](https://help.github.com/articles/creating-a-pull-request/)


### Short for time?

Feel free to raise an [issue](https://github.com/bfortuner/ml-glossary/issues) to correct errors or contribute content without a pull request.


## Style Guide

Each entry in the glossary MUST include the following at a minimum:

1. **Concise explanation** - as short as possible, but no shorter
2. **Citations** - Papers, Tutorials, etc.

Excellent entries will also include:

1. **Visuals** - diagrams, charts, animations, images
2. **Code** - python/numpy snippets, classes, or functions
3. **Equations** - Formatted with Latex

The goal of the glossary is to present content in the most accessible way possible, with a heavy emphasis on visuals and interactive diagrams. That said, in the spirit of rapid prototyping, it's okay to to submit a "rough draft" without visuals or code. We expect other readers will enhance your submission over time.


## Why RST and not Markdown?

RST has more features. For large and complex documentation projects, it's the logical choice.

* https://eli.thegreenplace.net/2017/restructuredtext-vs-markdown-for-technical-documentation/


## Top Contributors

We're big fans of [Distill](http://distill.pub/prize) and we like their idea of offering prizes for high-quality submissions. We don't have as much money as they do, but we'd still like to reward contributors in some way for contributing to the glossary. For instance a cheatsheet cryptocurreny where tokens equal commits ;). Let us know if you have better ideas. In the end, this is an open-source project and we hope contributing to a repository of concise, accessible, machine learning knowledge is enough incentive on its own!


## Tips and Tricks

* [Adding equations](http://www.sphinx-doc.org/en/stable/ext/math.html)
* [Working with Jupyter Notebook](http://louistiao.me/posts/demos/ipython-notebook-demo/)
* Quickstart with Jupyter notebook template
* Graphs and charts
* Importing images
* Linking to code


## Resources

* [Desmos Graphing Tool](https://www.desmos.com/calculator)
* [3D Graphing Tool](https://www.geogebra.org/3d)
* [How To Submit Pull Requests](https://help.github.com/articles/creating-a-pull-request/)
* [RST Cheatsheet](http://docutils.sourceforge.net/docs/user/rst/quickref.html)
* [Markdown Cheatsheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet)
* [Citation Generator](http://www.citationmachine.net)
* [MathJax Cheatsheet](https://math.meta.stackexchange.com/questions/5020/mathjax-basic-tutorial-and-quick-reference)
* [Embedding Math Equations](http://www.sphinx-doc.org/en/stable/ext/math.html)
* [Sphinx Tutorial](https://pythonhosted.org/an_example_pypi_project/sphinx.html)
* [Sphinx Docs](http://www.sphinx-doc.org/en/stable/markup/code.html)
* [Sphinx Cheatsheet](http://openalea.gforge.inria.fr/doc/openalea/doc/_build/html/source/sphinx/rest_syntax.html)
