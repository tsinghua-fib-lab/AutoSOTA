from typing import Optional

from appl import AIRole, SystemMessage, gen, ppl, records


class Agent(object):
    def __init__(self, name: Optional[str] = None):
        self._name = name
        self._history = self._setup()  # initialize history

    @ppl
    def _setup(self):
        if self._name:
            SystemMessage(f"Your name is {self._name}.")
        return records()

    @ppl
    def chat(self, msg):
        self._history  # retrieve history
        msg  # add to the prompt
        with AIRole():
            (reply := gen(max_tokens=50))  # generate reply and add to the prompt
        self._history = records()  # update history
        return reply


alice = Agent("Alice")
bob = Agent("Bob")
msg = "Hello!"
for i in range(2):
    msg = str(alice.chat(msg))
    print("Alice:", msg)
    msg = str(bob.chat(msg))
    print("Bob:", msg)
