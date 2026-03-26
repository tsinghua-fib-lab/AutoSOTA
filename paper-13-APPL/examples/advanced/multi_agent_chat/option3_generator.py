from typing import Optional

from appl import AIRole, SystemMessage, gen, ppl, records


class Agent(object):
    def __init__(self, name: Optional[str] = None):
        self._name = name
        self.chat = self._chat_generator().send

    @ppl
    def _setup(self):
        if self._name:
            SystemMessage(f"Your name is {self._name}.")
        return records()

    @ppl(auto_prime=True)  # auto prime the generator
    def _chat_generator(self):
        self._setup()
        reply = None
        while True:
            yield reply  # yield and receive messages
            with AIRole():
                (reply := gen(max_tokens=50))


alice = Agent("Alice")
bob = Agent("Bob")
msg = "Hello!"
for i in range(2):
    msg = str(alice.chat(msg))
    print("Alice:", msg)
    msg = str(bob.chat(msg))
    print("Bob:", msg)
