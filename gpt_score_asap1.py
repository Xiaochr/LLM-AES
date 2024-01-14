import pandas as pd 
import openai
import time
import re
from dataset_info import *
from sklearn.model_selection import train_test_split


# proxies = {
#     "http": "127.0.0.1:2022", 
#     "https": "127.0.0.1:2022"
# }
# openai.proxy = proxies

# input your openai api_key
openai.api_key = ""


def zeroshot_worubrics_prompt(essay_prompt, essay):
    """
        zero-shot without rubrics setting
    """
    prompt = """
As a virtual evaluator with expertise in English composition, your role is to critically analyze and grade student essays. You are to act as an impartial judge and evaluate the essays based on the quality of the writing and adherence to the essay prompt.

Scores should be assigned on a scale of 1 to 6, where 1 represents poor quality, and 6 represents excellent quality. 

Sample Essay Prompt: 
{}

Student's Essay to Evaluate
{}

Please present your evaluation in the following list format:
[Score: ..., Explanations: ...]
""".format(essay_prompt, essay)

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
    """
        zero-shot with rubrics setting
    """
    prompt = """
As a virtual evaluator with expertise in English composition, your role is to critically analyze and grade student essays according to a predetermined set of rubrics. You are to act as an impartial judge and evaluate the essays based on the quality of the writing and adherence to the essay prompt.

Scores should be assigned on a scale of 1 to 6, where 1 represents poor quality, and 6 represents excellent quality. 

Here are the specific guidelines for each score:
{}

Sample Essay Prompt: 
{}

Student's Essay to Evaluate
{}

Please present your evaluation in the following list format:
[Score: ..., Explanations: ...]
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
    """
        few-shot with rubrics setting
    """ 
    prompt = """
As a virtual evaluator with expertise in English composition, your role is to critically analyze and grade student essays according to a predetermined set of rubrics and graded examples. You are to act as an impartial judge and evaluate the essays based on the quality of the writing and adherence to the essay prompt.

Scores should be assigned on a scale of 1 to 6, where 1 represents poor quality, and 6 represents excellent quality. 

Here are the specific guidelines for each score:
{}

Sample Essay Prompt: 
{}

The graded example essays:
{}

Student's Essay to Evaluate
{}

Please present your evaluation in the following list format:
[Score: ..., Explanations: ...]
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


def get_gpt_score(essay_set=1, prompt_func=zeroshot_worubrics_prompt):
    data = pd.read_excel("./ASAP/training_set_rel3.xlsx")
    data = data[data["essay_set"] == essay_set]
    data.index = range(len(data))
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)

    data = test_df
    data.index = range(len(data))
    print(len(data))
    data["essay"] = data["essay"].apply(text_preprocessing)
    print(data.head())

    score_list = []
    cnt = 0
    for essay in data["essay"]:
        print(cnt)
        flag = 0
        while flag == 0:
            try:
                temp_score = prompt_func(set_1_rubrics, set_1_prompt, set_1_examples, essay)
                # temp_score = prompt_func(set_1_rubrics, set_1_prompt, essay)
                # temp_score = prompt_func(set_1_prompt, essay)
                flag = 1
            except:
                time.sleep(60)
                flag = 0
                print("error")
        
        score_list.append(temp_score)
        
        if cnt == 0:
            print(temp_score)

        cnt += 1
    
    data["output"] = score_list
    print(data.head())
    
    data.to_csv("./your_output_file_name.csv")



def text_preprocessing(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)
    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)
    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text




if __name__ == '__main__':
    get_gpt_score(essay_set=1, prompt_func=fewshot_rubrics_prompt)




