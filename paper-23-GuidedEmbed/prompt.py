text_summarize_prompt = """
Summarize the text based on the following instruction. The summary must focus on the instruction's key points and not exceed 10 words.

Instruction: {instruction}

Text:
{text}

Required format:
Summary: <your_summary>

Note: Only output the summary in English starting with "Summary:", do not include any other text.
"""

label_generation_prompt = """
Analyze these two groups of texts and define a clear category label that best describes the characteristics of the current group based on the following instruction.

Instruction: {instruction}

Current Group Texts:
--------------------------------------------------
{positive_texts}
--------------------------------------------------

Other Group Texts:
--------------------------------------------------
{negative_texts}
--------------------------------------------------

Requirements:
- The label MUST strictly follow and reflect the given instruction
- Focus on the main characteristics of the current group based on the instruction
- Label should be generalizable but distinguishable from other texts
- Use clear and precise language
- The category name should be no more than 5 words

Required format:
Category: <category_name>

Note: Only output the category name starting with "Category:", do not include any other text.
"""

text_classification_prompt = """
Please classify the following text based on the instruction and available categories.

Instruction: {instruction}

Available Categories:
{categories}

Text to classify:
{text}

Required format:
Classification: <category>

Note: Only output the category name starting with "Classification:", do not include any other text.
The category must be exactly as listed above.
"""