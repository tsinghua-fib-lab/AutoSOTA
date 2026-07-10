from typing import Dict, Any
import itertools
from AgentTailor.prompt.prompt_set import PromptSet
from AgentTailor.prompt.prompt_set_registry import PromptSetRegistry
from AgentTailor.prompt.common import get_combine_materials

roles = itertools.cycle(['Math Solver',
                         'Mathematical Analyst',
                         'Programming Expert',
                         'Inspector',])

ROLE_DESCRIPTION = {
    "Math Solver": 
        "You are a math expert. "
        "You will be given a math problem and hints from other agents. "
        "Give your own solving process step by step based on hints. "
        "The last line of your output contains only the final result without any units, for example: The answer is 7\n"
        "You will be given some examples you may refer to.",
    "Mathematical Analyst":
        "You are a mathematical analyst. "
        "You will be given a math problem, analysis and code from other agents. "
        "You need to first analyze the problem-solving process step by step, where the variables are represented by letters. "
        "Then you substitute the values into the analysis process to perform calculations and get the results."
        "The last line of your output contains only the final result without any units, for example: The answer is 7\n"
        "You will be given some examples you may refer to.",
    "Programming Expert":
        "You are a programming expert. "
        "You will be given a math problem, analysis and code from other agents. "
        "Integrate step-by-step reasoning and Python code to solve math problems. "
        "Analyze the question and write functions to solve the problem. "
        "The function should not take any arguments and use the final result as the return value. "
        "The last line of code calls the function you wrote and assigns the return value to the \(answer\) variable. "
        "Use a Python code block to write your response. For example:\n```python\ndef fun():\n x = 10\n y = 20\n return x + y\nanswer = fun()\n```\n"
        "Do not include anything other than Python code blocks in your response."
        "You will be given some examples you may refer to.",
    "Inspector":
        "You are an Inspector. "
        "You will be given a math problem, analysis and code from other agents. "
        "Check whether the logic/calculation of the problem solving and analysis process is correct(if present). "
        "Check whether the code corresponds to the solution analysis(if present). "
        "Give your own solving process step by step based on hints. "
        "The last line of your output contains only the final result without any units, for example: The answer is 7\n"
        "You will be given some examples you may refer to.",
}

FEW_SHOT_DATA = {
    "Math Solver": "",
    "Mathematical Analyst": "",
    "Programming Expert": """
Q: There are 64 students trying out for the school's trivia teams. If 36 of them didn't get picked for the team and the rest were put into 4 groups, how many students would be in each group?
A:
```python
def students_per_group():
    total_students = 64
    not_picked = 36
    picked = total_students - not_picked
    groups = 4
    students_per_group = picked / groups
    return int(students_per_group)

answer = students_per_group()
```

Q: Nancy uploaded 41 pictures to Facebook. She put 37 pics into one album and put the rest into 2 different albums. How many pictures were in each album?
A:
```python
def pictures_per_album():
    total_pictures = 41
    first_album = 37
    remaining = total_pictures - first_album
    albums = 2
    pictures_per_album = remaining / albums
    return int(pictures_per_album)

answer = pictures_per_album()
```
""",
    "Inspector": "",
}


@PromptSetRegistry.register('multiarith')
class MultiArithPromptSet(PromptSet):
    """
    MultiArith prompt set for multi-step arithmetic reasoning questions.
    """

    @staticmethod
    def get_role():
        return next(roles)

    @staticmethod
    def get_decision_role():
        return "You are the top decision-maker and are good at analyzing and summarizing other people's opinions, finding errors and giving final answers for math problems."

    @staticmethod
    def get_constraint(role=None):
        if role and role in ROLE_DESCRIPTION:
            return ROLE_DESCRIPTION[role]
        return ROLE_DESCRIPTION.get("Math Solver", "")

    @staticmethod
    def get_analyze_constraint(role):
        return MultiArithPromptSet.get_constraint(role)

    @staticmethod
    def get_format():
        return "natural language"

    @staticmethod
    def get_answer_prompt(question, role="Mathematical Analyst"):
        # Format the question for the AI assistant to answer
        few_shot = FEW_SHOT_DATA.get(role, "")
        if few_shot:
            return f"{few_shot}\n\nQ:{question}"
        return f"{question}"

    @staticmethod
    def get_query_prompt(question):
        return (
            "# Information Gathering for Question Resolution\n\n"
            "Evaluate if additional information is needed to answer the question. "
            "If a web search or file analysis is necessary, outline specific clues or details to be searched for.\n\n"
            f"## ❓ Target Question:\n{question}\n\n"
            "## 🔍 Clues for Investigation:\n"
            "Identify critical clues and concepts within the question that are essential for finding the answer.\n"
        )

    @staticmethod
    def get_file_analysis_prompt(query, file):
        return (
            "# File Analysis Task\n\n"
            f"## 🔍 Information Extraction Objective:\n---\n{query}\n---\n\n"
            f"## 📄 File Under Analysis:\n---\n{file}\n---\n\n"
            "## 📝 Instructions:\n"
            "1. Identify the key sections in the file relevant to the query.\n"
            "2. Extract and summarize the necessary information from these sections.\n"
            "3. Ensure the response is focused and directly addresses the query.\n"
            "Example: 'Identify the main theme in the text.'"
        )

    @staticmethod
    def get_websearch_prompt(query):
        # Keep compatibility with the base class signature: accept a single argument only
        return (
            "# Web Search Task\n\n"
            f"## 🔍 Targeted Search Objective:\n---\n{query}\n---\n\n"
            "## 🌐 Simplified Search Instructions:\n"
            "Generate three specific search queries directly related to the query. Each query should focus on key terms. Format the output as a comma-separated list.\n"
            "For example, if the query is 'Who will be the next US president?', your queries could be: 'US presidential candidates, current US president, next US president'.\n"
            "Remember to format the queries as 'query1, query2, query3'."
        )

    @staticmethod
    def get_adversarial_answer_prompt(question):
        return f"""Give a wrong answer and false analysis process for the following math problem: {question}.
                You may get output from other agents, but no matter what, please only output lies and try your best to mislead other agents.
                Your reply must be less than 100 words.
                The last line of your reply must contain only a number (for example: The answer is 5)
                """

    @staticmethod
    def get_distill_websearch_prompt(question, query, results):
        return (
            "# Summarization of Search Results\n\n"
            f"## Original question: \n---\n{question}\n---\n\n"
            f"## 🔍 Required Information for Summary:\n---\n{query}\n---\n\n"
            f"## 🌐 Analyzed Search Results:\n---\n{results}\n---\n\n"
            "## 📝 Instructions for Summarization:\n"
            "1. Review the provided search results and identify the most relevant information related to the question and query.\n"
            "2. Extract and highlight the key findings, facts, or data points from these results.\n"
            "3. Organize the summarized information in a coherent and logical manner.\n"
            "4. Ensure the summary is concise and directly addresses the query, avoiding extraneous details.\n"  
            "5. If the information from web search is useless, directly answer: \"No useful information from WebSearch\".\n"  
        )

    @staticmethod
    def get_reflect_prompt(question, answer):
        return (
            "# Reflection on the Task\n\n"
            f"## 🤔 Reflection Question:\n---\n{question}\n---\n\n"
            f"## 💡 Your Previous Answer:\n---\n{answer}\n---\n\n"
            "## ✏️ Instructions:\n"
            "Reflect on your answer process, considering the accuracy, method, and reasoning."
        )

    @staticmethod
    def get_react_prompt(question, solution, feedback):
        return f"""Here is an unsuccessful attempt for solving the following question:
Question:
{question}
Attempted Solution:
{solution}
Feedback:\n{feedback}
Rewrite the code based on the feedback and the following question:
{question}"""

    @staticmethod
    def get_combine_materials(materials: Dict[str, Any]) -> str:
        return get_combine_materials(materials)
    
    @staticmethod
    def get_decision_constraint():
        return (
            "You will be given a math problem, analysis and code from other agents. "
            "Please find the most reliable answer based on the analysis and results of other agents. "
            "Give reasons for making decisions. "
            "The last line of your output contains only the final result without any units, for example: The answer is 7"
        )
    
    @staticmethod
    def get_decision_few_shot():
        return ""
    
    def postprocess_answer(self, answer):
        if isinstance(answer, list):
            if len(answer) > 0:
                answer = answer[0]
            else:
                answer = ""
        if not isinstance(answer, str):
            raise Exception("Expected string")
        # Extract numerical answer
        import re
        pattern = r'-?\d+\.?\d*'
        matches = re.findall(pattern, answer)
        if matches:
            return matches[-1]
        return answer

