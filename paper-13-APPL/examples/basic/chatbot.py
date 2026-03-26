from appl import AIRole, SystemMessage, gen, ppl


@ppl
def chat1(name: str):
    # choice1: all-in-one chat function
    SystemMessage(f"Your name is {name}")

    while True:
        print("User:")
        (msg := input())
        if msg.startswith("exit"):
            break
        print("Assistant:")
        with AIRole():
            (reply := gen(stream=True))
        print(reply)


# For the first call, the passed-in context is stored
# For following calls, stored context is resumed
@ppl(ctx="resume")
def bot(msg: str):
    msg
    with AIRole():
        (reply := gen(stream=True))
    return reply


@ppl
def chat2(name: str):
    # choice2: call separate function to chat
    # Prepare the context
    SystemMessage(f"Your name is {name}")

    while True:
        print("User:")
        msg = input()
        if msg.startswith("exit"):
            break
        print("Assistant:")
        print(bot(msg))


@ppl(auto_prime=True)
def chat_gen(name: str):
    SystemMessage(f"Your name is {name}")
    reply = None
    while True:
        yield reply
        with AIRole():
            (reply := gen())


def chat3(name: str):
    # choice3: use generator
    chat = chat_gen(name)
    while True:
        print("User:")
        msg = input()
        if msg.startswith("exit"):
            break
        print("Assistant:")
        print(chat.send(msg))


if __name__ == "__main__":
    # chat1("Alice")
    chat2("Bob")
    # chat3("Chelsea")
