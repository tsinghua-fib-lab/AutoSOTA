class PromptTemplate:
    def __init__(self, system_prompt=None):
        self.system_prompt = system_prompt
        self.user_messages = []
        self.model_replies = []

    def __str__(self):
        return self.build_prompt()

    def add_user_message(self, message: str, return_prompt=True):
        self.user_messages.append(message)
        if return_prompt:
            return self.build_prompt()

    def add_model_reply(self, reply: str, includes_history=True, return_reply=False):
        reply_ = reply.replace(self.build_prompt(), "") if includes_history else reply
        self.model_replies.append(reply_)
        if len(self.user_messages) != len(self.model_replies):
            raise ValueError(
                "Number of user messages does not equal number of system replies."
            )
        if return_reply:
            return reply_

    def get_user_messages(self, strip=True):
        return [x.strip() for x in self.user_messages] if strip else self.user_messages

    def get_model_replies(self, strip=True):
        return [x.strip() for x in self.model_replies] if strip else self.model_replies

    def build_prompt(self):
        if len(self.user_messages) != len(self.model_replies) + 1:
            raise ValueError(
                "Error: Expected len(user_messages) = len(model_replies) + 1. Add a new user message!"
            )

        if self.system_prompt is not None:
            SYS = f"[INST] <<SYS>>\n{self.system_prompt}\n<</SYS>> "
        else:
            SYS = "[INST]"

        CONVO = ""
        SYS = "<s> " + SYS
        for i in range(len(self.user_messages) - 1):
            user_message, model_reply = self.user_messages[i], self.model_replies[i]
            conversation_ = f"{user_message} [/INST] {model_reply} </s> "
            if i != 0:
                conversation_ = "<s> [INST] " + conversation_
            CONVO += conversation_

        if len(self.user_messages)==1:
          CONVO += f"{self.user_messages[-1]} [/INST]"
        else:
          CONVO += f"<s> [INST] {self.user_messages[-1]} [/INST]"

        return SYS + CONVO