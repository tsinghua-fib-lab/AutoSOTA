import logging
from src.defense import RRAG

# 提供若干基线防御/聚合策略的实现类（继承自 src.defense.RRAG），
# 每个类通过构造特定的 prompt 把 top-k 检索内容与问题送入后端 LLM，得到最终回答并返回字符串。
# 主要用于和仓库中更复杂的防御方法做对比。

logger = logging.getLogger('RRAG-main')

class InstructRAG_ICL(RRAG):
    """
        InstructRAG-ICL
        Source: https://github.com/HuichiZhou/TrustRAG/blob/main/defend_module.py
        todo: sampling_params
    """
    def __init__(
        self, 
        llm,
    ):
        super().__init__(llm)

    def query(self, data_item):
        question = data_item["question"]
        top_k = data_item["topk_content"]
        document_list = ""
        for index, doc in enumerate(top_k):
            document_list += f"Document[{index}]: " + doc + "\n"

        if 'choices' in data_item:
            icl_prompt = f"""
                Your task is to analyze the provided documents and answer the given question. Please generate a brief explanation of how the contents of these documents lead to your answer. If the provided information is not helpful to answer the question, you only need to respond based on your own knowledge, without referring to the documents.

                Below are some examples of how to answer the question:

                ###

                Example 1

                Question: where did the term christian name come from?

                Answer: The documents that are useful to answer the question "where did the term 'Christian name' come from?" are Documents 1, 2, and 5.

                Document 1 provides the etymology of the term "Christian", which comes from the Greek word "Christos", meaning "anointed one". This sets the foundation for understanding the origin of the term.

                Document 2 explains that the term "Christian" was used as a baptismal name, and that it has been used as a given name since the Middle Ages.

                Document 5 provides the definition of "Christian name" as "baptismal name", and explains that the term "christen" means "to baptize". This document also mentions the practice of giving a name to a male child at the time of circumcision, which is relevant to the origin of the term "Christian name".

                Based on these documents, it can be deduced that the term "Christian name" originated from the practice of giving a name to a person at the time of baptism, which was a common practice in the early Christian Church. This practice was likely influenced by the Hebrew tradition of giving a name to a male child at the time of circumcision.

                The answer "Apostolic origin" is not explicitly supported by the provided documents, but it can be inferred that the practice of giving a name at baptism was likely influenced by the early Christian Church, which was founded by the apostles. Therefore, the term "Christian name" can be said to have originated from the Apostolic era, although the exact origin is not explicitly stated in the provided documents.

                In conclusion, the term "Christian name" originated from the practice of giving a name to a person at the time of baptism, which was influenced by the early Christian Church and the Hebrew tradition of giving a name at circumcision.

                ###

                Example 2

                Question: the process in which the value of ∆ u = 0 is?

                Answer: After reviewing the provided documents, I found that Documents 1, 4, and 5 are relevant to answering the question "the process in which the value of ∆u = 0 is?".

                Document 1 defines the Gibbs free energy (ΔG) as the maximum amount of non-expansion work that can be extracted from a thermodynamically closed system at a constant temperature and pressure. It also mentions that the Gibbs free energy is related to the internal energy (U) as ΔG = ΔH - TΔS, where H is the enthalpy and S is the entropy.

                Document 4 defines an isothermal process as a change of a system where the temperature remains constant (ΔT = 0). This process typically occurs when a system is in contact with an outside thermal reservoir and the change in the system occurs slowly enough to allow the system to adjust to the temperature of the reservoir through heat exchange.

                Document 5 discusses thermodynamic equilibrium, which is characterized by the free energy being at its minimum value. The free energy change (δG) can be expressed as a weighted sum of chemical potentials, which are related to the partial molar free energies of the species in equilibrium.

                To answer the question, we can analyze the relationship between the Gibbs free energy (ΔG) and the internal energy (U). In an isothermal process, the temperature remains constant (ΔT = 0), which means that the entropy (S) remains constant. Therefore, the change in internal energy (ΔU) can be related to the change in Gibbs free energy (ΔG) as:

                ΔU = ΔG + PΔV

                where P is the pressure and V is the volume.

                Since the process is isothermal, the pressure and volume are constant, and the change in internal energy (ΔU) is equal to the change in Gibbs free energy (ΔG). Therefore, in an isothermal process, the value of ΔU = 0 when the value of ΔG = 0.

                In conclusion, the process in which the value of ∆u = 0 is an isothermal process, as it is the only process where the temperature remains constant, and the change in internal energy (ΔU) is equal to the change in Gibbs free energy (ΔG).
                
                ###

                Now it is your turn to analyze the following documents and answer the given question. Highlight your exact final answer within the tag <ANSWER> your final answer </ANSWER>.

                {document_list}

                Based on your knowledge and the provided information, find the best candidate answer to the question: {question}

                Candidates: \nA. {data_item.get("choices")[0]} \nB. {data_item.get("choices")[1]} \nC. {data_item.get("choices")[2]} \nD. {data_item.get("choices")[3]} \n
                
                Answer: \n
                """
        else:
            icl_prompt = f"""
                Your task is to analyze the provided documents and answer the given question. Please generate a brief explanation of how the contents of these documents lead to your answer. If the provided information is not helpful to answer the question, you only need to respond based on your own knowledge, without referring to the documents.

                Below are some examples of how to answer the question:

                ###

                Example 1

                Question: where did the term christian name come from?

                Answer: The documents that are useful to answer the question "where did the term 'Christian name' come from?" are Documents 1, 2, and 5.

                Document 1 provides the etymology of the term "Christian", which comes from the Greek word "Christos", meaning "anointed one". This sets the foundation for understanding the origin of the term.

                Document 2 explains that the term "Christian" was used as a baptismal name, and that it has been used as a given name since the Middle Ages.

                Document 5 provides the definition of "Christian name" as "baptismal name", and explains that the term "christen" means "to baptize". This document also mentions the practice of giving a name to a male child at the time of circumcision, which is relevant to the origin of the term "Christian name".

                Based on these documents, it can be deduced that the term "Christian name" originated from the practice of giving a name to a person at the time of baptism, which was a common practice in the early Christian Church. This practice was likely influenced by the Hebrew tradition of giving a name to a male child at the time of circumcision.

                The answer "Apostolic origin" is not explicitly supported by the provided documents, but it can be inferred that the practice of giving a name at baptism was likely influenced by the early Christian Church, which was founded by the apostles. Therefore, the term "Christian name" can be said to have originated from the Apostolic era, although the exact origin is not explicitly stated in the provided documents.

                In conclusion, the term "Christian name" originated from the practice of giving a name to a person at the time of baptism, which was influenced by the early Christian Church and the Hebrew tradition of giving a name at circumcision.

                ###

                Example 2

                Question: the process in which the value of ∆ u = 0 is?

                Answer: After reviewing the provided documents, I found that Documents 1, 4, and 5 are relevant to answering the question "the process in which the value of ∆u = 0 is?".

                Document 1 defines the Gibbs free energy (ΔG) as the maximum amount of non-expansion work that can be extracted from a thermodynamically closed system at a constant temperature and pressure. It also mentions that the Gibbs free energy is related to the internal energy (U) as ΔG = ΔH - TΔS, where H is the enthalpy and S is the entropy.

                Document 4 defines an isothermal process as a change of a system where the temperature remains constant (ΔT = 0). This process typically occurs when a system is in contact with an outside thermal reservoir and the change in the system occurs slowly enough to allow the system to adjust to the temperature of the reservoir through heat exchange.

                Document 5 discusses thermodynamic equilibrium, which is characterized by the free energy being at its minimum value. The free energy change (δG) can be expressed as a weighted sum of chemical potentials, which are related to the partial molar free energies of the species in equilibrium.

                To answer the question, we can analyze the relationship between the Gibbs free energy (ΔG) and the internal energy (U). In an isothermal process, the temperature remains constant (ΔT = 0), which means that the entropy (S) remains constant. Therefore, the change in internal energy (ΔU) can be related to the change in Gibbs free energy (ΔG) as:

                ΔU = ΔG + PΔV

                where P is the pressure and V is the volume.

                Since the process is isothermal, the pressure and volume are constant, and the change in internal energy (ΔU) is equal to the change in Gibbs free energy (ΔG). Therefore, in an isothermal process, the value of ΔU = 0 when the value of ΔG = 0.

                In conclusion, the process in which the value of ∆u = 0 is an isothermal process, as it is the only process where the temperature remains constant, and the change in internal energy (ΔU) is equal to the change in Gibbs free energy (ΔG).
                
                ###

                Now it is your turn to analyze the following documents and answer the given question. Highlight your exact final answer within the tag <ANSWER> your final answer </ANSWER>.

                {document_list}

                Based on your knowledge and the provided information, answer the question: {question}
                """
        response = self.llm.query(icl_prompt)

        logger.debug(f'Query_prompt:\n{icl_prompt}')
        logger.debug(f'Response:\n{response}')
        logger.debug(f'Answer:\n{data_item["answer"]}')
        return response


class AstuteRAG(RRAG):
    """
        AstuteRAG
        Source: https://github.com/HuichiZhou/TrustRAG/blob/main/defend_module.py
        todo: sampling_params
        todo: t>1 not implemented yet (iterative consolidation)
    """
    def __init__(
        self, 
        llm,
    ):
        super().__init__(llm)

    def query(self, data_item):
        question = data_item["question"]
        top_k = data_item["topk_content"]
        document_list = ""
        for index, doc in enumerate(top_k):
            document_list += f"Externally Retrieved Document[{index}]: " + doc + "\n"

        stage_one_prompt = f"""Generate a document that provides accurate and relevant information to answer the given question. If the information is unclear or uncertain, explicitly state 'I don't know' to avoid any hallucinations.
        Question: {question} 
        Document:"""

        internal_knowledge = self.llm.query(stage_one_prompt)

        document_list = document_list + "\n" + f"Memorized Document: " + internal_knowledge + "\n"

        if 'choices' in data_item:
            final_prompt = f"""Task: Answer a given question using the consolidated information from both your own
            memorized documents and externally retrieved documents.
            Step 1: Consolidate information
            * For documents that provide consistent information, cluster them together and summarize
            the key details into a single, concise document.
            * For documents with conflicting information, separate them into distinct documents, ensuring
            each captures the unique perspective or data.
            * Exclude any information irrelevant to the query. For each new document created, clearly indicate:
            * Whether the source was from memory or an external retrieval. * The original document numbers for transparency.
            Step 2: Propose Answers and Assign Confidence
            For each group of documents, propose a possible answer and assign a confidence score based on the credibility and agreement of the information.
            Step 3: Select the Final Answer
            After evaluating all groups, select the most accurate and well-supported answer from the set of candidate answers. Highlight your exact final answer within the tag <ANSWER> your answer </ANSWER>.
            Initial Context: {document_list}
            Question: {question}
            Candidates: \nA. {data_item.get("choices")[0]} \nB. {data_item.get("choices")[1]} \nC. {data_item.get("choices")[2]} \nD. {data_item.get("choices")[3]} \n
            Answer: \n
            """
        else:
            final_prompt = f"""Task: Answer a given question using the consolidated information from both your own
            memorized documents and externally retrieved documents.
            Step 1: Consolidate information
            * For documents that provide consistent information, cluster them together and summarize
            the key details into a single, concise document.
            * For documents with conflicting information, separate them into distinct documents, ensuring
            each captures the unique perspective or data.
            * Exclude any information irrelevant to the query. For each new document created, clearly indicate:
            * Whether the source was from memory or an external retrieval. * The original document numbers for transparency.
            Step 2: Propose Answers and Assign Confidence
            For each group of documents, propose a possible answer and assign a confidence score based on the credibility and agreement of the information.
            Step 3: Select the Final Answer
            After evaluating all groups, select the most accurate and well-supported answer. Highlight your exact final answer within the tag <ANSWER> your answer </ANSWER>.
            Initial Context: {document_list}
            Question: {question}
            Answer:
            """            

        response = self.llm.query(final_prompt)
        logger.debug(f'Stage one prompt:\n{stage_one_prompt}')
        logger.debug(f'Query_prompt:\n{final_prompt}')
        logger.debug(f'Response:\n{response}')
        logger.debug(f'Answer:\n{data_item["answer"]}')
        return response

class TrustRAG(RRAG):
    def __init__(
        self, 
        llm,
    ):
        super().__init__(llm)

    def query(self, data_item):
        pass
