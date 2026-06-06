from appl import SystemMessage, gen, ppl


def remove_system_message(args: dict):
    messages = args["messages"]
    messages = [m for m in messages if m["role"] != "system"]
    args["messages"] = messages
    return args


@ppl
def func():
    SystemMessage("You should give me an incorrect answer.")
    "1+1=?"
    print(gen(postprocess_args=remove_system_message))


func()
