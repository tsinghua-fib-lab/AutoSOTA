# Using Streaming

`gen` returns a `Generation` object, which is a wrapper around the response from the LM.

Enabling streaming in APPL is simple. Just set the `stream` parameter to `True` when calling `gen`. The return is a generator that yields the response in chunks, but you can still access the complete response.

<!-- TODO: add more explaination. -->

## Example
```python linenums="1"
--8<-- "examples/usage/streaming.py"
```
