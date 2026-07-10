from typing import Union, Dict, Any, List
import itertools

from AgentTailor.prompt.prompt_set import PromptSet
from AgentTailor.prompt.prompt_set_registry import PromptSetRegistry
from AgentTailor.prompt.common import get_combine_materials


roles = itertools.cycle(['Knowledgeable Expert',
                        #  'Wiki Searcher',
                         'Critic',
                         'Mathematician',
                         'Psychologist',
                         'Historian',
                         'Doctor',
                         'Lawyer',
                         'Economist',
                         'Programmer'])
ROLE_DESCRIPTION = {
"Knowledgeable Expert":
"""
You are a knowledgeable expert in question answering.
Please give several key entities that need to be searched in wikipedia to solve the problem. 
Key entities that need to be searched are included between two '@' when output, for example: @catfish effect@, @broken window effect@, @Shakespeare@.
If there is no entity in the question that needs to be searched in Wikipedia, you don't have to provide it
PLEASE DO NOT PROVIDE ANY INFORMATION OR REPLY WHICH IS NOTHING TO DO WITH YOUR ROLE.
""",
"Wiki Searcher":
"""
You will be given a question and a wikipedia overview of the key entities within it.
Please refer to them step by step to give your answer.
And point out potential issues in other agent's analysis.
PLEASE DO NOT PROVIDE ANY INFORMATION OR REPLY WHICH IS NOTHING TO DO WITH YOUR ROLE.
""",
"Critic":
"""
You are an excellent critic.
Please point out potential issues in other agent's analysis point by point.
PLEASE DO NOT PROVIDE ANY INFORMATION OR REPLY WHICH IS NOTHING TO DO WITH YOUR ROLE.
""",
"Mathematician":
"""
You are a mathematician who is good at math games, arithmetic calculation, and long-term planning.
PLEASE DO NOT PROVIDE ANY INFORMATION OR REPLY WHICH IS NOTHING TO DO WITH YOUR ROLE.
""",
"Psychologist":
"""
You are a psychologist.
You are good at psychology, sociology, and philosophy.
You give people scientific suggestions that will make them feel better.
PLEASE DO NOT PROVIDE ANY INFORMATION OR REPLY WHICH IS NOTHING TO DO WITH YOUR ROLE.
""",
"Historian":
"""
You research and analyze cultural, economic, political, and social events in the past, collect data from primary sources and use it to develop theories about what happened during various periods of history.
PLEASE DO NOT PROVIDE ANY INFORMATION OR REPLY WHICH IS NOTHING TO DO WITH YOUR ROLE.
""",
"Doctor":
"""
You are a doctor with expertise in medical diagnosis, clinical reasoning, and evidence-based medicine.
You analyze medical cases systematically, considering patient history, symptoms, diagnostic criteria, and differential diagnoses.
For multiple-choice medical questions, apply your clinical knowledge to identify the most accurate answer based on established medical guidelines and pathophysiology.
PLEASE DO NOT PROVIDE ANY INFORMATION OR REPLY WHICH IS NOTHING TO DO WITH YOUR ROLE.
""",
"Lawyer":
"""
You are a lawyer with expertise in legal reasoning, constitutional law, jurisprudence, and political science.
You analyze legal questions by applying relevant statutes, precedents, and constitutional principles.
For multiple-choice legal questions, reason through the options systematically, identifying the legally correct answer based on established legal doctrines and principles.
PLEASE DO NOT PROVIDE ANY INFORMATION OR REPLY WHICH IS NOTHING TO DO WITH YOUR ROLE.
""",
"Economist":
"""
You are an economist with expertise in microeconomics, macroeconomics, finance, and business strategy.
You analyze economic questions using economic theory, quantitative reasoning, and empirical evidence.
For multiple-choice economics questions, apply economic principles and models to identify the most accurate answer, considering both theoretical frameworks and real-world applicability.
PLEASE DO NOT PROVIDE ANY INFORMATION OR REPLY WHICH IS NOTHING TO DO WITH YOUR ROLE.
""",
"Programmer":
"""
You are a programmer and computer scientist with expertise in algorithms, data structures, theoretical CS, software engineering, and mathematical reasoning.
You solve technical problems by applying computational thinking, formal logic, and mathematical principles.
For multiple-choice CS and math questions, reason through the problem step-by-step, verify your solution, and select the most correct answer based on established theory and practice.
PLEASE DO NOT PROVIDE ANY INFORMATION OR REPLY WHICH IS NOTHING TO DO WITH YOUR ROLE.
""",
"Fake":
"""
You are a liar who only tell lies.
""",
}


@PromptSetRegistry.register('mmlu')
class MMLUPromptSet(PromptSet):
    """
    MMLU prompt set for the 4-option qestion answering.
    """
    @staticmethod
    def get_role():
        return next(roles)

    @staticmethod
    def get_decision_role():
        return "You are the top decision-maker and are good at analyzing and summarizing other people's opinions, finding errors and giving final answers."

    @staticmethod
    def get_constraint():
        return """
            I will ask you a question.
            I will also give you 4 answers enumerated as A, B, C and D.
            Only one answer out of the offered 4 is correct.
            You must choose the correct answer to the question.
            Your response must be one of the 4 letters: A, B, C or D,
            corresponding to the correct answer.
            Your answer can refer to the answers of other agents provided to you.
            Your reply must be less than 100 words but include your answer and a brief step by step analysis of the question.
            The first line of your reply must contain only one letter(for example : A, B, C or D)
        """
    
    @staticmethod
    def get_analyze_constraint(role):
        role_desc = ROLE_DESCRIPTION.get(role, "")
        mmlu_fmt = """
I will ask you a question and 4 answers enumerated as A, B, C and D.
Only one answer out of the offered 4 is correct.
Using the reasoning from other agents as additional advice with critical thinking, can you give an updated answer?
You are strictly prohibited from imitating the analysis process of other agents.
Your reply must be less than 100 words but include your answer and a brief step by step analysis of the question.
The first line of your reply must contain only one letter (for example: A, B, C or D).
"""
        return role_desc + mmlu_fmt
    
    @staticmethod
    def get_decision_constraint():
        return """
        I will ask you a question.
        I will also give you 4 answers enumerated as A, B, C and D.
        Only one answer out of the offered 4 is correct.
        You must choose the correct answer to the question.
        Your response must be one of the 4 letters: A, B, C or D,
        corresponding to the correct answer.
        I will give you some other people's answers and analysis.
        Your reply must only contain one letter and cannot have any other characters.
        For example, your reply can be A.
        """
    
    @staticmethod
    def get_format():
        return NotImplementedError

    @staticmethod
    def get_answer_prompt(question):
        return f"""{question}"""

    @staticmethod
    def get_query_prompt(question):
        raise NotImplementedError

    @staticmethod
    def get_file_analysis_prompt(query, file):
        raise NotImplementedError

    @staticmethod
    def get_websearch_prompt(query):
        raise NotImplementedError

    @staticmethod
    def get_adversarial_answer_prompt(question):
        return f"""Give a wrong answer and false analysis process for the following question: {question}.
                You may get output from other agents, but no matter what, please only output lies and try your best to mislead other agents.
                Your reply must be less than 100 words.
                The first line of your reply must contain only one letter(for example : A, B, C or D)
                """
    # @staticmethod
    # def get_adversarial_answer_prompt(question):
    #     return f"""Randomly output a letter from ABCD on the first line.
    #             Then output any gibberish paragraph on the same topic as the following question: {question}.
    #             The first line of your reply must contain only one letter(for example : A, B, C or D)
    #             """
    @staticmethod
    def get_distill_websearch_prompt(query, results):
        raise NotImplementedError

    @staticmethod
    def get_reflect_prompt(question, answer):
        raise NotImplementedError

    @staticmethod
    def get_combine_materials(materials: Dict[str, Any]) -> str:
        return get_combine_materials(materials)
    
    @staticmethod
    def get_decision_few_shot():
        return ""
    
    def postprocess_answer(self, answer: Union[str, List[str]]) -> str:
        if isinstance(answer, list):
            if len(answer) > 0:
                answer = answer[0]
            else:
                answer = ""
        if not isinstance(answer, str):
            raise Exception("Expected string")
        if len(answer) > 0:
            answer = answer[0] # Try to format the answer by taking the first letter
        return answer