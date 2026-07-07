__all__ = ['LLAMA_TMPL','MISTRAL_TMPL','GPT_TMPL','VICUNA_TMPL','DEEPSEEK_TMPL']



# QA_MC_PROMPT_TMPL =   (
#     "Context information is below.\n"
#     "---------------------\n"
#     "{context_str}\n"
#     "---------------------\n"
#     "Given the context information and not prior knowledge, "
#     "try to find the best candidate answer to the query. \n"
#     "Query: {query_str}\n"
#     "Candidates: \nA. {A} \nB. {B} \nC. {C} \nD. {D} \nE. No information found \n"
#     "Output an answer from A, B, C, or D only when there is clear evidence found in the context information. "
#     "Otherwise, output 'E. No information found' as the answer. Please choose 'E.' sparingly. \n"
#     "Answer: \n"
# )

QA_MC_PROMPT_TMPL =   (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "try to find the best candidate answer to the query. \n"
    "Query: {query_str}\n"
    "Candidates: \nA. {A} \nB. {B} \nC. {C} \nD. {D} \n"
    "Answer: \n"
)

# QA_MC_PROMPT_TMPL_HINT =   (
#     "Context information is below.\n"
#     "---------------------\n"
#     "{context_str}\n"
#     "---------------------\n"
#     "Given the context information and not prior knowledge, "
#     "try to find the best candidate answer to the query. \n"
#     "Query: {query_str}\n"
#     "Candidates: \nA. {A} \nB. {B} \nC. {C} \nD. {D} \nE. No information found \n"
#     "Some hints are in the word list provided below.\n The answer should appear in the word list."
#     "Word list: \n"
#     "---------------------\n"
#     "{hints}\n"
#     "---------------------\n"
#     "Output an answer from A, B, C, or D only when there is clear evidence found in the context information. "
#     "Otherwise, output 'E. No information found' as the answer. Please choose 'E.' sparingly.\n"
#     "Answer: \n"
# )

QA_MC_PROMPT_TMPL_HINT =   (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "try to find the best candidate answer to the query. \n"
    "Query: {query_str}\n"
    "Candidates: \nA. {A} \nB. {B} \nC. {C} \nD. {D} \n"
    "Some hints are in the word list provided below.\n The answer should appear in the word list."
    "Word list: \n"
    "---------------------\n"
    "{hints}\n"
    "---------------------\n"
    "Answer: \n"
)


QA_MC_PROMPT_TMPL_LLAMA =   (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "try to find the best candidate answer to the query. \n"
    "Query: {query_str}\n"
    "Candidates: \nA. {A} \nB. {B} \nC. {C} \nD. {D} \nE. No information found \n"
    "Answer: \n"
)

QA_MC_PROMPT_TMPL_VICUNA =   (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
    "USER: "
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "try to find the best candidate answer to the query. \n"
    "Query: {query_str}\n"
    "Candidates: \nA. {A} \nB. {B} \nC. {C} \nD. {D} \nE. No information found \n"
    # "Output an answer from A, B, C, or D only when there is clear evidence found in the context information. "
    # "Otherwise, output 'E. No information found' as the answer. \n"
    "ASSISTANT: The answer is: \n"
)

ZERO_QA_MC_PROMPT_TMPL =  (
    "Answer the query with the best candidates. \n"
    #"If you cannot find the answer, just say \"I don\'t know\".\n"
    "Query: {query_str}\n"
    "Candidates: \nA. {A} \nB. {B} \nC. {C} \nD. {D} \nE. No information found \n"
    "Output an answer from A, B, C, or D only when there is clear evidence. "
    "Otherwise, output 'E. No information found' as the answer. \n" # TODO: add "do not guess the answer"?
    "Answer: \n"
)



QA_PROMPT_TMPL =   (

    "Context information is below.\n"
    "---------------------\n"
    '''NASA's Artemis Program Advances
    In 2022, NASA made significant progress in the Artemis program, aimed at returning humans to the Moon and establishing a sustainable presence by the end of the decade... \n'''
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query with only keywords. \n"
    "If there is no relevant information, just say \"I don\'t know\".\n" 
    "Query: What is the primary goal of NASA's Artemis program?\n"
    "Answer: Return humans to the Moon\n\n\n"

    "Context information is below.\n"
    "---------------------\n"
    '''2022 US Women's Open Highlights
    The 2022 US Women’s Open was concluded in June at Pine Needles Lodge & Golf Club in North Carolina. Minjee Lee emerged victorious capturing ... \n'''
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query with only keywords. \n"
    "If there is no relevant information, just say \"I don\'t know\".\n" 
    "Query: Which golfer won the 2022 US Women’s Open?\n"
    "Answer: Minjee Lee\n\n\n"
    
    "Context information is below.\n"
    "---------------------\n"
    '''Microsoft acquires gaming company
    Microsoft has completed the acquisition of the gaming company Activision Blizzard. This move is expected to enhance Microsoft's gaming portfolio and significantly boost its market share in the gaming industry...\n'''
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query with only keywords. \n"
    "If there is no relevant information, just say \"I don\'t know\".\n" 
    "Query: What new video game titles are being released by Microsoft this year?\n"
    "Answer: I don't know\n\n\n"

    "Context information is below.\n"
    "---------------------\n"
    '''Apple launches iPhone 14 with satellite connectivity
    Apple has officially launched the iPhone 14, which includes a groundbreaking satellite connectivity feature for emergency situations. This feature is designed to ensure safety in remote areas without cellular service...\n'''
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query with only keywords. \n"
    "If there is no relevant information, just say \"I don\'t know\".\n" 
    "Query: What new feature does the iPhone 14 have?\n"
    "Answer: Satellite connectivity\n\n\n"

    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query with only keywords. \n"
    "If there is no relevant information, just say \"I don\'t know\".\n" 
    "Query: {query_str}\n"
    "Answer: "
)

QA_PROMPT_TMPL_VICUNA =   (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query with no more than ten words. \n"
    "If there is no relevant information, just say \"I don\'t know\".\n" 
    "Query: {query_str}\n"
    "Answer: \n"
)


ZERO_QA_PROMPT_TMPL =  (
    #"Answer the query within five words. \n"
    "Answer the query with no more than ten words. \n"
    "If you do not know the answer confidently, just say \"I don\'t know\".\n"
    "Query: {query_str}\n"
    "Answer: \n"
)

ZERO_QA_DECODE_PROMPT_TMPL =  (
    "Write an accurate, engaging, and concise answer. \n"
    "If there is no relevant information, just say \"I don\'t know\".\n" 
    "Query: What is the primary goal of NASA's Artemis program?\n"
    "Answer: Return humans to the Moon\n\n\n"

    "Write an accurate, engaging, and concise answer. \n"
    "If there is no relevant information, just say \"I don\'t know\".\n" 
    "Query: Which golfer won the 2022 US Women’s Open?\n"
    "Answer: Minjee Lee\n\n\n"
    
    "Write an accurate, engaging, and concise answer. \n"
    "If there is no relevant information, just say \"I don\'t know\".\n" 
    "Query: What new video game titles are being released by Microsoft this year?\n"
    "Answer: I don't know\n\n\n"

    "Write an accurate, engaging, and concise answer. \n"
    "If there is no relevant information, just say \"I don\'t know\".\n" 
    "Query: What new feature does the iPhone 14 have?\n"
    "Answer: Satellite connectivity\n\n\n"

    "Write an accurate, engaging, and concise answer. \n"
    "If there is no relevant information, just say \"I don\'t know\".\n" 
    "Query: {query_str}\n"
    "Answer: "
)


# QA_PROMPT_HINTS_TMPL =  (
#     "Word suggestion is below.\n"
#     "---------------------\n"
#     "{hints}\n"
#     "---------------------\n"
#     "Given the word suggestion and not prior knowledge, answer the query with no more than ten words.\n"
#     #"Use words from suggestion above and answer the query with no more than ten words.\n"
#     "Query: {query_str}\n"
#     "Answer: \n"
# ) 


QA_PROMPT_HINTS_TMPL =  (
    "Word suggestion is below.\n"
    "---------------------\n"
    "Starfield, Halo, 2023 releases, Forza\n"
    "---------------------\n"
    "Given the word suggestion provided by experts, concisely answer the query.\n"
    "Query: What new video game titles are being released by Microsoft this year?\n"
    "Answer: Starfield, Forza Motorsport\n\n\n"

    "Word suggestion is below.\n"
    "---------------------\n"
    "California, $15.50, Minimum wage, Increased, Economic impact\n"
    "---------------------\n"
    "Given the word suggestion provided by experts, concisely answer the query.\n"
    "Query: What is the minimum wage in California as of this year?\n"
    "Answer: $15.50\n\n\n"

    "Word suggestion is below.\n"
    "---------------------\n"
    "Olympic gold, Swimming, Phelps, Michael\n"
    "---------------------\n"
    "Given the word suggestion provided by experts, concisely answer the query.\n"
    "Query: Who holds the record for the most Olympic gold medals in swimming?\n"
    "Answer: Michael Phelps\n\n\n"

    "Word suggestion is below.\n"
    "---------------------\n"
    "NASA, James Webb Telescope, Space exploration, $1 billion, Funding\n"
    "---------------------\n"
    "Given the word suggestion provided by experts, concisely answer the query.\n"
    "Query:  How much did NASA invest in developing the James Webb Space Telescope?\n"
    "Answer: $1 billion or $1 billion funding\n\n\n"

    "Word suggestion is below.\n"
    "---------------------\n"
    "{hints}\n"
    "---------------------\n"
    "Given the word suggestion provided by experts, concisely answer the query.\n"
    "Query: {query_str}\n"
    "Answer: "
) 







QA_PROMPT_LONG_TMPL =   (
    "Context information is below.\n"
    "---------------------\n"
    '''Albert Einstein was born on March 14, 1879, in Ulm, Germany, and later became one of the most famous physicists in history. Known for his mass–energy equivalence formula E = mc^2, which has been dubbed "the world's most famous equation," he significantly contributed to modern physics theories. Besides his scientific achievements, Einstein had a passion for music, particularly loving the violin. In 1921, he was awarded the Nobel Prize in Physics, primarily for his explanation of the photoelectric effect, rather than his relativity theories. Einstein also held outspoken political views, advocating for civil rights and socialism, and was a member of the NAACP. He expressed concerns about the rise of fascism in Europe in the 1930s. Einstein also enjoyed sailing and was known for not being very good at it. He moved to the United States in the early 1930s, where he worked at the Institute for Advanced Study in Princeton, New Jersey. He died there in 1955. During his life, he owned several pet cats, one of which was named Tiger.'''
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "write an accurate, engaging, and concise answer.\n"
    "If there is no relevant information, just say \"I don\'t know\".\n"
    "Query: Tell me a bio of Albert Einstein? \n"
    "Answer: Albert Einstein, born in 1879 in Ulm, Germany, was a physicist famed for his theory of relativity and the iconic equation E = mc^2, illustrating the equivalence of mass and energy. Awarded the Nobel Prize in Physics in 1921 for his explanation of the photoelectric effect, Einstein's contributions extended beyond physics, including advocacy for civil rights and anti-fascism. An avid music and sailing enthusiast, he spent his later years in Princeton, New Jersey, at the Institute for Advanced Study, passing away in 1955. His personal life included a passion for violin and his pet cat, Tiger, reflecting his multifaceted character.\n\n#####\n\n"

    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "write an accurate, engaging, and concise answer.\n"
    "If there is no relevant information, just say \"I don\'t know\".\n"
    "Query: {query_str}\n"
    "Answer: \n"
)

QA_PROMPT_LONG_ZERO_TMPL =  (
    "write an accurate, engaging, and concise answer.\n"
    "If you do not know the answer confidently, just say \"I don\'t know\".\n"
    "Query: Tell me a bio of Albert Einstein? \n"
    "Answer: Albert Einstein, born in 1879 in Ulm, Germany, was a physicist famed for his theory of relativity and the iconic equation E = mc^2, illustrating the equivalence of mass and energy. Awarded the Nobel Prize in Physics in 1921 for his explanation of the photoelectric effect, Einstein's contributions extended beyond physics, including advocacy for civil rights and anti-fascism. An avid music and sailing enthusiast, he spent his later years in Princeton, New Jersey, at the Institute for Advanced Study, passing away in 1955. His personal life included a passion for violin and his pet cat, Tiger, reflecting his multifaceted character.\n\n#####\n\n"

    "Write an accurate, engaging, and concise answer.\n"
    "If you do not know the answer confidently, just say \"I don\'t know\".\n"
    "Query: {query_str}\n"
    "Answer: \n"
)

QA_PROMPT_LONG_GEN_HINTS = ( 
    "Context information is below.\n"
    "---------------------\n"
    '''Albert Einstein was born on March 14, 1879, in Ulm, Germany, and later became one of the most famous physicists in history. Known for his mass–energy equivalence formula E = mc^2, which has been dubbed "the world's most famous equation," he significantly contributed to modern physics theories. Besides his scientific achievements, Einstein had a passion for music, particularly loving the violin. In 1921, he was awarded the Nobel Prize in Physics, primarily for his explanation of the photoelectric effect, rather than his relativity theories. Einstein also held outspoken political views, advocating for civil rights and socialism, and was a member of the NAACP. He expressed concerns about the rise of fascism in Europe in the 1930s. Einstein also enjoyed sailing and was known for not being very good at it. He moved to the United States in the early 1930s, where he worked at the Institute for Advanced Study in Princeton, New Jersey. He died there in 1955. During his life, he owned several pet cats, one of which was named Tiger.'''
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "Extract a few important short important phrases from it to facilitate the query. \n"
    "Query: Tell me a bio of Albert Einstein? \n"
    "Answer: Albert Einstein, Born 1879, Ulm Germany, Nobel Prize, Physics Professor, Physics 1921, Relativity theories, Photoelectric effect, violin music, Civil rights, IAS Princeton\n\n\n"
    
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "Extract a few important short important phrases from it to facilitate the query. \n"
    "Query: {query_str} \n"
    "Answer: \n")

QA_PROMPT_LONG_HINTS = ( 
    "Write an accurate, engaging, and concise answer.\n"
    "Query: Tell me a bio of Albert Einstein? \n"
    "Answer the above question with the following important phrases suggestions: [Born 1879, Ulm Germany, Nobel Prize, Albert Einstein, Physics Professor, Physics 1921, photoelectric effect, violin music, civil rights, IAS Princeton] "
    "The final answer is: Albert Einstein, born in 1879 in Ulm, Germany, was a physicist famed for his theory of relativity and the iconic equation E = mc^2, illustrating the equivalence of mass and energy. Awarded the Nobel Prize in Physics in 1921 for his explanation of the photoelectric effect, Einstein's contributions extended beyond physics, including advocacy for civil rights and anti-fascism. An avid music and sailing enthusiast, he spent his later years in Princeton, New Jersey, at the Institute for Advanced Study, passing away in 1955. His personal life included a passion for violin and his pet cat, Tiger, reflecting his multifaceted character.\n\n\n"

    "Write an accurate, engaging, and concise answer.\n"
    "Query: {query_str}\n"
    "Answer the above question with the following important phrases suggestions: [{hints}] "
    "The final answer is: \n"
    )



#MULTIPLE_PROMPT_HINTS = '''You are a helpful assistant, Below, you will find the user's question accompanied by a series of hints. These hints are the aggregated outputs from various experts, ordered by their frequency. Your answer should be informative. \
#\n\nQuery: [question]  \nAnswer the above question with the following word suggestions: [hints] '''

DEEPSEEK_TMPL = {
    'qa-mc': QA_MC_PROMPT_TMPL_LLAMA,
    'qa-mc-zero': ZERO_QA_MC_PROMPT_TMPL, 
    'qa-mc-hint': QA_MC_PROMPT_TMPL_HINT,

    'qa': QA_PROMPT_TMPL,
    'qa-zero': ZERO_QA_PROMPT_TMPL,
    'qa-zero-decode': ZERO_QA_DECODE_PROMPT_TMPL,
    'qa-hint':QA_PROMPT_HINTS_TMPL,

    'qa-long':QA_PROMPT_LONG_TMPL,
    'qa-long-zero':QA_PROMPT_LONG_ZERO_TMPL,
    'qa-long-genhint-hint':QA_PROMPT_LONG_HINTS,
    'qa-long-genhint':QA_PROMPT_LONG_GEN_HINTS,
}

LLAMA_TMPL = {
    'qa-mc': QA_MC_PROMPT_TMPL_LLAMA,
    'qa-mc-zero': ZERO_QA_MC_PROMPT_TMPL, 
    'qa-mc-hint': QA_MC_PROMPT_TMPL_HINT,

    'qa': QA_PROMPT_TMPL,
    'qa-zero': ZERO_QA_PROMPT_TMPL,
    'qa-zero-decode': ZERO_QA_DECODE_PROMPT_TMPL,
    'qa-hint':QA_PROMPT_HINTS_TMPL,

    'qa-long':QA_PROMPT_LONG_TMPL,
    'qa-long-zero':QA_PROMPT_LONG_ZERO_TMPL,
    'qa-long-genhint-hint':QA_PROMPT_LONG_HINTS,
    'qa-long-genhint':QA_PROMPT_LONG_GEN_HINTS,
}


MISTRAL_TMPL = {
    'qa-mc': QA_MC_PROMPT_TMPL,
    'qa-mc-zero': ZERO_QA_MC_PROMPT_TMPL, 
    'qa-mc-hint': QA_MC_PROMPT_TMPL_HINT,

    'qa': QA_PROMPT_TMPL,
    'qa-zero': ZERO_QA_PROMPT_TMPL,
    'qa-zero-decode': ZERO_QA_DECODE_PROMPT_TMPL,
    'qa-hint':QA_PROMPT_HINTS_TMPL,

    'qa-long':QA_PROMPT_LONG_TMPL,
    'qa-long-zero':QA_PROMPT_LONG_ZERO_TMPL,
    'qa-long-genhint-hint':QA_PROMPT_LONG_HINTS,
    'qa-long-genhint':QA_PROMPT_LONG_GEN_HINTS,
}

VICUNA_TMPL = {
    'qa-mc': QA_MC_PROMPT_TMPL_VICUNA,
    'qa-mc-zero': ZERO_QA_MC_PROMPT_TMPL, 
    'qa-mc-hint': QA_MC_PROMPT_TMPL_HINT,

    'qa': QA_PROMPT_TMPL_VICUNA,
    'qa-zero': ZERO_QA_PROMPT_TMPL,
    'qa-zero-decode': ZERO_QA_DECODE_PROMPT_TMPL,
    'qa-hint':QA_PROMPT_HINTS_TMPL,

    'qa-long':QA_PROMPT_LONG_TMPL,
    'qa-long-zero':QA_PROMPT_LONG_ZERO_TMPL,
    'qa-long-genhint-hint':QA_PROMPT_LONG_HINTS,
    'qa-long-genhint':QA_PROMPT_LONG_GEN_HINTS,
}








QA_PROMPT_HINTS_TMPL_GPT =  (
    "Answer the query question using only words from the word list provided below.\n"
    "Query: {query_str}\n"
    "Word list: \n"
    "---------------------\n"
    "{hints}\n"
    "---------------------\n"
    "Answer: \n"
) 

QA_PROMPT_TMPL_GPT =   (
    "Please answer the query question based on the update-to-date context information provided below.\n"
    "Query: {query_str}\n"
    "Context:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    '''It is very important that the answer should be based solely on evidence found in the context information. Answer concisely. If the context is completely irrelevant, say "I don't know"\n'''
    "Answer: \n"
)


QA_PROMPT_LONG_HINTS_GPT= ( 

    "Here is an example for the task.\n"
    "Write an accurate, engaging, and concise answer.\n"
    "Query: Tell me a bio of Albert Einstein? \n"
    "Answer the above question with the following important phrases suggestions: [Born 1879, Ulm Germany, Nobel Prize, Albert Einstein, Physics Professor, Physics 1921, photoelectric effect, violin music, civil rights, IAS Princeton] "
    "The final answer is: Albert Einstein, born in 1879 in Ulm, Germany, was a physicist famed for his theory of relativity and the iconic equation E = mc^2, illustrating the equivalence of mass and energy. Awarded the Nobel Prize in Physics in 1921 for his explanation of the photoelectric effect, Einstein's contributions extended beyond physics, including advocacy for civil rights and anti-fascism. An avid music and sailing enthusiast, he spent his later years in Princeton, New Jersey, at the Institute for Advanced Study, passing away in 1955. His personal life included a passion for violin and his pet cat, Tiger, reflecting his multifaceted character.\n\n\n"

    "Here is the final task.\n"
    "Write an accurate, engaging, and concise answer.\n"
    "Query: {query_str}\n"
    "Answer the above question with the following important phrases suggestions: [{hints}] "
    "The final answer is: \n"
    )

QA_PROMPT_LONG_GEN_HINTS_GPT = ( 
    "Here is an example for the task.\n"
    "Context information is below.\n"
    "---------------------\n"
    '''Albert Einstein was born on March 14, 1879, in Ulm, Germany, and later became one of the most famous physicists in history. Known for his mass–energy equivalence formula E = mc^2, which has been dubbed "the world's most famous equation," he significantly contributed to modern physics theories. Besides his scientific achievements, Einstein had a passion for music, particularly loving the violin. In 1921, he was awarded the Nobel Prize in Physics, primarily for his explanation of the photoelectric effect, rather than his relativity theories. Einstein also held outspoken political views, advocating for civil rights and socialism, and was a member of the NAACP. He expressed concerns about the rise of fascism in Europe in the 1930s. Einstein also enjoyed sailing and was known for not being very good at it. He moved to the United States in the early 1930s, where he worked at the Institute for Advanced Study in Princeton, New Jersey. He died there in 1955. During his life, he owned several pet cats, one of which was named Tiger.'''
    "---------------------\n"
    "Given the context information, write an accurate, engaging, and concise answer. \n"
    "Query: Tell me a bio of Albert Einstein? \n"
    "Answer:  Albert Einstein, born in 1879 in Ulm, Germany, was a physicist famed for his theory of relativity and the iconic equation E = mc^2, illustrating the equivalence of mass and energy. Awarded the Nobel Prize in Physics in 1921 for his explanation of the photoelectric effect, Einstein's contributions extended beyond physics, including advocacy for civil rights and anti-fascism. An avid music and sailing enthusiast, he spent his later years in Princeton, New Jersey, at the Institute for Advanced Study, passing away in 1955. His personal life included a passion for violin and his pet cat, Tiger, reflecting his multifaceted character."
    
    "Here is the final task.\n"
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information, write an accurate, engaging, and concise answer. \n"
    "Query: {query_str} \n"

    "Answer: \n")



GPT_TMPL = {
    'qa-mc': QA_MC_PROMPT_TMPL,
    'qa-mc-zero': ZERO_QA_MC_PROMPT_TMPL, 
    'qa-mc-hint': QA_MC_PROMPT_TMPL_HINT,

    'qa': QA_PROMPT_TMPL_GPT,
    'qa-zero': ZERO_QA_PROMPT_TMPL,
    'qa-hint':QA_PROMPT_HINTS_TMPL_GPT,

    'qa-long':QA_PROMPT_LONG_TMPL,
    'qa-long-zero':QA_PROMPT_LONG_ZERO_TMPL,
    'qa-long-genhint-hint':QA_PROMPT_LONG_HINTS_GPT,
    'qa-long-genhint':QA_PROMPT_LONG_GEN_HINTS_GPT,
}
