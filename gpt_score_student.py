import pandas as pd 
import numpy as np 
import openai
import os 
import time
import re
from dataset_info import *


# proxies = {
#     "http": "127.0.0.1:2022", 
#     "https": "127.0.0.1:2022"
# }
# openai.proxy = proxies


openai.api_key = ""



student_prompt = """
Suppose you are Li Hua, a senior student at Hongxing High School. Your school is currently soliciting ideas for the senior graduation ceremony. You are interested in participating and have formed some ideas about the event design. Please write an email to your British friend Jim asking for his advice. The content should include:
1. Introduce your design ideas;
2. Explain the reasons behind your design.
"""

student_prompt_2 = """
Suppose you are Li Hua, a senior high school student at Hongxing High School. Recently, you received a letter from your British friend Jim, learning that his plan to go on a bike trip with friends for a week did not get his parents' permission, and he feels very disappointed. Please write a reply to Jim, including:
1. Expressing consolation;
2. Offering advice.
"""



def zeroshot_worubrics_prompt(essay_prompt, essay):    
    prompt = """
As a virtual evaluator with expertise in English composition, your role is to critically analyze and grade student essays. You are to act as an impartial judge and evaluate the essays based on the quality of the writing and adherence to the essay prompt.

Three dimensions of scores and a total score should be assigned, where higher scores represent higher quality: 
1. Content (on a scale of 0 to 8)
2. Language (on a scale of 0 to 8)
3. Structure (on a scale of 0 to 4)
4. Total Score (on a scale of 0 to 20) = Content Score + Language Score + Structure Score

Sample Essay Prompt: 
{}

Student's Essay to Evaluate
{}

Please present your evaluation in the following manner:

Explanations: ...
Content Score: ...

Explanations: ...
Language Score: ...

Explanations: ...
Structure Score: ...

Explanations: ...
Total Score: ...

Your final evaluation: 
[Total Score: ..., Content Score: ..., Language Score: ..., Structure Score: ...]""".format(essay_prompt, essay)

    response = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0.0,
        messages=[
                {"role": "system", "content": "As a virtual evaluator with expertise in English composition, your role is to critically analyze and grade student essays."},
                {"role": "user", "content": prompt}
            ]
    )

    response_content = response["choices"][0]["message"]["content"]
    return response_content


def zeroshot_rubrics_prompt(rubrics, essay_prompt, essay):    
    prompt = """
As a virtual evaluator with expertise in English composition, your role is to critically analyze and grade student essays according to a predetermined set of rubrics. You are to act as an impartial judge and evaluate the essays based on the quality of the writing and adherence to the essay prompt.

Three dimensions of scores and a total score should be assigned, where higher scores represent higher quality: 
1. Content (on a scale of 0 to 8)
2. Language (on a scale of 0 to 8)
3. Structure (on a scale of 0 to 4)
4. Total Score (on a scale of 0 to 20) = Content Score + Language Score + Structure Score

Here are the specific guidelines for each score:
{}

Sample Essay Prompt: 
{}

Student's Essay to Evaluate: 
{}

Task Breakdown:
1. Carefully read the provided essay prompt, scoring guidelines, and the student's essay.
2. In the Explanations part, identifying specific elements in the essay referring to the rubrics. In the language dimension, list all the spelling and grammar errors, and count the number of them to determine the Language Score. The Explanations for each dimension should be as detailed as possible.
3. Determine the appropraite scores according to the analysis above. 

Please present your evaluation in the following manner:

Explanations: ...
Content Score: ...

Explanations: ...
Language Score: ...

Explanations: ...
Structure Score: ...

Explanations: ...
Total Score: ...

Your final evaluation: 
[Total Score: ..., Content Score: ..., Language Score: ..., Structure Score: ...]
""".format(rubrics, essay_prompt, essay)

    response = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0.0,
        messages=[
                {"role": "system", "content": "As a virtual evaluator with expertise in English composition, your role is to critically analyze and grade student essays according to a predetermined set of rubrics."},
                {"role": "user", "content": prompt}
            ]
    )

    response_content = response["choices"][0]["message"]["content"]
    return response_content




def fewshot_rubrics_prompt(rubrics, essay_prompt, examples, essay):    
    prompt = """
As a virtual evaluator with expertise in English composition, your role is to critically analyze and grade student essays according to a predetermined set of rubrics and graded examples. You are to act as an impartial judge and evaluate the essays based on the quality of the writing and adherence to the essay prompt.

Three dimensions of scores and a total score should be assigned, where higher scores represent higher quality: 
1. Content (on a scale of 0 to 8)
2. Language (on a scale of 0 to 8)
3. Structure (on a scale of 0 to 4)
4. Total Score (on a scale of 0 to 20) = Content Score + Language Score + Structure Score

Here are the specific guidelines for each score:
{}

Sample Essay Prompt: 
{}

The graded example essays:
{}

Student's Essay to Evaluate: 
{}

Task Breakdown:
1. Carefully read the provided essay prompt, scoring guidelines, and the student's essay.
2. In the Explanations part, identifying specific elements in the essay referring to the rubrics. In the language dimension, list all the spelling and grammar errors, and count the number of them to determine the Language Score. The Explanations for each dimension should be as detailed as possible.
3. Determine the appropraite scores according to the analysis above. 

Please present your evaluation in the following manner:

Explanations: ...
Content Score: ...

Explanations: ...
Language Score: ...

Explanations: ...
Structure Score: ...

Explanations: ...
Total Score: ...

Your final evaluation: 
[Total Score: ..., Content Score: ..., Language Score: ..., Structure Score: ...]
""".format(rubrics, essay_prompt, examples, essay)

    response = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0.0,
        messages=[
                {"role": "system", "content": "As a virtual evaluator with expertise in English composition, your role is to critically analyze and grade student essays according to a predetermined set of rubrics and graded examples."},
                {"role": "user", "content": prompt}
            ]
    )

    response_content = response["choices"][0]["message"]["content"]
    return response_content




