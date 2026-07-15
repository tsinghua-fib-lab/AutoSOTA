# Documentation

> Good documentation is very important.

We make and update the documentation based on [Sphinx](https://www.sphinx-doc.org) with some support of Markdown.
So we can use both reStructuredText and Markdown files.

## Required packages

Required packages should be installed by

```shell
pip install "appfl[dev]" --upgrade
```

We use `nbsphinx` to add Jupyter notebooks to the doc, which requires `pandoc` package. This package may need to be manually installed from conda.

```shell
conda install -c conda-forge pandoc
```

## Build document

The build command for documentation is:

```shell
cd appfl/docs
make html
```

Note that `vscode` can preview the docs.
