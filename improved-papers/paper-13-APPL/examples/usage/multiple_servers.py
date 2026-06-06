from appl import gen, ppl


@ppl
def add1(a, b):
    f"what's {a} plus {b}?"
    return gen("azure-gpt35")


@ppl
def add2(a, b):
    f"what's {a} plus {b}?"
    return gen("claude-35-sonnet")


@ppl
def add3(a, b):
    f"what's {a} plus {b}?"
    return gen("moonshot-8k")


@ppl
def add4(a, b):
    f"what's {a} plus {b}?"
    return gen("srt-llama2")


# print(add1("one", "two"))
# print(add2("three", "four"))
# print(add3("five", "six"))
# print(add4("seven", "eight"))
