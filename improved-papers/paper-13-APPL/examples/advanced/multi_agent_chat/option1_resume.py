from typing import Optional

from appl import AIRole, SystemMessage, gen, ppl, traceable


class Agent(object):
    def __init__(self, model_name: str, name: Optional[str] = None):
        self._model_name = model_name
        self._name = name
        self._setup()

    @ppl
    def _setup(self):
        if self._name:
            SystemMessage(f"Your name is {self._name}.")
        self.chat(None)  # setup the context

    @ppl(ctx="resume")  # The context is resumed from the last call to chat
    def chat(self, msg: Optional[str]):
        if msg is None:  # first call to setup the context
            return
        msg  # add to the prompt
        with AIRole():
            (reply := gen(self._model_name, max_tokens=50))
            # generate reply and add to the prompt
        return reply


@traceable
def main():
    alice = Agent("gpt-4o", "Alice")
    bob = Agent("gpt-4o-mini", "Bob")
    msg = "Hello"
    for i in range(2):
        msg = str(alice.chat(msg))
        print("Alice:", msg)
        msg = str(bob.chat(msg))
        print("Bob:", msg)


if __name__ == "__main__":
    main()
