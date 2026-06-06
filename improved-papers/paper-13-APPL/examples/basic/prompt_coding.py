from appl import BracketedDefinition as Def
from appl import define, ppl, records
from appl.compositor import *
from appl.const import EMPTY


# declare the input and output requirements classes for reference
class InputReq(Def):
    name = "Input Requirement"


class OutputReq(Def):
    name = "Output Requirement"


# Alternatively, but not recommended, you can define classes as follows:
# InputReq = define("Input Requirement")
# OutputReq = define("Output Requirement")


@ppl
def get_prompt(opr: str, language: str):
    "## Requirements"
    with NumberedList():
        # instantiate the input requirement
        InputReq(desc="The input should be two numbers.")
        # instantiate the output requirement
        OutputReq(desc=f"The output should be the {opr} of the two numbers.")
    EMPTY
    "## Instruction"
    # The naming can be used to distinguish:
    # variable naming (e.g. language): the dynamic input
    # class naming (e.g. InputReq): the reference to the concept
    f"Write a function in {language} that satisfies the {InputReq} and {OutputReq}."
    return records()


print("=== First prompt ===")
print(get_prompt("sum", "Python"))
print("")
print("=== Second prompt ===")
print(get_prompt("product", "C++"))
