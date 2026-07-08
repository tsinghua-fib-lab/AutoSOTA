def gsm_prompt(doc):
  system_prompt = (
      "You are a math expert. You will be given a question to solve. Solve it step by step. Wrap the final answer in a \\boxed{}. \n"
      "Respond in the following format:\n"
      "<reasoning>\n"
      "Your reasoning here\n"
      "</reasoning>\n"
      "<answer>\n"
      "\\boxed{...}\n"
      "</answer>"
  )
  prompt = f"{system_prompt}\n\n{doc['question']}\n\n"
  return prompt
