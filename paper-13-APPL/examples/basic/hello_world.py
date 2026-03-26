from appl import gen, ppl


@ppl  # the @ppl decorator marks the function as an `APPL function`
def greeting(name: str):
    f"Hello World! My name is {name}."  # Add text to the prompt
    return gen()  # call the default LLM with the current prompt


print(greeting("APPL"))  # call `hello_world` as a normal Python function
