from abc import ABC, abstractmethod
from typing import Optional

from appl import AIRole, PromptContext, SystemMessage, gen, ppl
from appl.func import wraps


class AgentBase(ABC):
    def __init__(self):
        self._ctx = PromptContext()  # the context for the agent
        self._setup(_ctx=self._ctx)  # manually provide the shared context

    @abstractmethod
    @ppl(ctx="same")
    def _setup(self):
        raise NotImplementedError

    @abstractmethod
    @ppl(ctx="same")
    def _chat(self, msg: str):
        raise NotImplementedError

    @wraps(_chat)
    def chat(self, *args, **kwargs):
        # manually provide the shared context self._ctx stored in the instance
        return self._chat(*args, _ctx=self._ctx, **kwargs)

    setattr(chat, "__isabstractmethod__", False)
    # set to False since _chat is abstractmethod


class Agent(AgentBase):
    def __init__(self, name: Optional[str] = None):
        self._name = name
        super().__init__()

    @ppl(ctx="same")
    def _setup(self):
        # modify the shared context
        if self._name:
            SystemMessage(f"Your name is {self._name}.")

    @ppl(ctx="same")
    def _chat(self, msg: str):
        msg
        with AIRole():
            (reply := gen(max_tokens=50))
        return reply


alice = Agent("Alice")
bob = Agent("Bob")
msg = "Hello!"
for i in range(2):
    msg = str(alice.chat(msg))
    print("Alice:", msg)
    msg = str(bob.chat(msg))
    print("Bob:", msg)
