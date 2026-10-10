# Contributing

Install the toolkit with `python -m pip install -e ".[dev]"` and run
`python -m unittest discover -s tests -v` before proposing a change.
Use synthetic test inputs. Describe changes to descriptor ordering, head settings,
metric interpolation, seeds, or ensemble behavior explicitly; these affect protocol fidelity.

Keep upstream code unchanged in `third_party/meisenmeister`. Put integration changes
in the toolkit. Do not commit clinical data, cached descriptors, weights, credentials,
or workstation-specific paths. Generated artifacts belong outside source control.

Format authored Python code with `python -m black src tests examples`.
The CI workflow checks formatting; the vendored baseline is deliberately excluded.
