import pandas as pd
import openai
import json 
import re 
from dataset_info import *
import time



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



def construct_dataset(train="test"):
    """
        dataset for fine-tuning the overall output model
    """
    data = pd.read_csv("./exam_{}.csv".format(train), index_col=0)
    print(data.head())

    template_list = []
    for i in range(len(data)):
        prompt = """
As a virtual evaluator with expertise in English composition, your role is to critically analyze and grade student essays according to a predetermined set of rubrics. You are to act as an impartial judge and evaluate the essays based on the quality of the writing and adherence to the essay prompt.

The total score should be assigned, where higher scores represent higher quality: 
1. Content (on a scale of 0 to 8)
2. Language (on a scale of 0 to 8)
3. Structure (on a scale of 0 to 4)
4. Total Score (on a scale of 0 to 20) = Content Score + Language Score + Structure Score

Here are the specific guidelines for each score:
{}

Sample Essay Prompt: 
{}

Student's Essay to Evaluate
{}

Please present your evaluation in the following list manner:
[Content Score, Language Score, Structure Score, Total Score]
For example: [6, 5, 3, 12]
""".format(true_rubrics, student_prompt, data["essay_content"][i])
        answer = str([data["content"][i], data["language"][i], data["structure"][i], data["total_score"][i]])

        template = {
            "messages": [
                {
                    "role": "system", 
                    "content": "As a virtual evaluator with expertise in English composition, your role is to critically analyze and grade student essays according to a predetermined set of rubrics."
                }, 
                {
                    "role": "user",
                    "content": prompt
                },
                {
                    "role": "assistant",
                    "content": answer
                }
            ]
        }
        template = json.dumps(template)
        template_list.append(template)

    with open("./exam_{}_json.jsonl".format(train), "w") as f:
        for temp in template_list:
            json.loads(temp)
            f.write(temp + "\n")



def construct_specific_dataset(score_type="total", train="test"):
    """
        dataset for fine-tuning specific output model (total, content, language, and structure)
    """
    data = pd.read_csv("./exam_{}.csv".format(train), index_col=0)
    print(data.head())

    template_list = []
    for i in range(len(data)):
        if score_type == "total":
            prompt = """
As a virtual evaluator with expertise in English composition, your role is to critically analyze and grade student essays according to a predetermined set of rubrics. You are to act as an impartial judge and evaluate the essays based on the quality of the writing and adherence to the essay prompt.

The total score should be assigned, where higher scores represent higher quality: 
1. Content (on a scale of 0 to 8)
2. Language (on a scale of 0 to 8)
3. Structure (on a scale of 0 to 4)
4. Total Score (on a scale of 0 to 20) = Content Score + Language Score + Structure Score

Here are the specific guidelines for each score:
{}

Sample Essay Prompt: 
{}

Student's Essay to Evaluate
{}

Please output the total score only. 
""".format(true_rubrics, student_prompt, data["essay_content"][i])
            answer = str(data["total_score"][i])
        
        elif score_type == "content":
            prompt = """
As a virtual evaluator with expertise in English composition, your role is to critically analyze and grade student essays according to a predetermined set of rubrics in Content dimension. You are to act as an impartial judge and evaluate the essays based on the quality of the writing and adherence to the essay prompt.

Here are the specific guidelines for the Content score:
Content Dimension (8 points in total)
- Level 1: 6-8 points:
    - Content is complete with appropriate details
    - Expression is closely related to the topic
- Level 2: 3-5 points:
    - Content is mostly complete
    - Expression is fundamentally related to the topic
- Level 3: 0-2 points:
    - Content is incomplete
    - Expression is barely related or completely unrelated to the topic

Sample Essay Prompt: 
{}

Student's Essay to Evaluate
{}

Please output the Content score only. 
""".format(student_prompt, data["essay_content"][i])
            answer = str(data["content"][i])

        elif score_type == "language":
            prompt = """
As a virtual evaluator with expertise in English composition, your role is to critically analyze and grade student essays according to a predetermined set of rubrics in Language dimension. You are to act as an impartial judge and evaluate the essays based on the quality of the writing and adherence to the essay prompt.

Here are the specific guidelines for the Language score:
Language Dimension (8 points in total)
- Level 1: 6-8 points:
    - Language is accurate with diverse sentence structures and little or no errors (2 errors or fewer, 8 points; 3-4 errors, 7 points; 5-6 errors, 6 points)
    - Language expression is mostly appropriate
- Level 2: 3-5 points:
    - Language is not quite accurate, with some variation in sentence structures and several errors, but they don't impede understanding (7-8 errors, 5 points; 9-10 errors, 4 points; 11-12 errors, 3 points)
    - Language expression is somewhat inappropriate
- Level 3: 0-2 points:
    - Language is hopelessly inaccurate with numerous language errors, hindering understanding (more than 12 errors)
    - Language expression is completely inappropriate

Sample Essay Prompt: 
{}

Student's Essay to Evaluate
{}

Please output the Language score only. 
""".format(student_prompt, data["essay_content"][i])
            answer = str(data["language"][i])

        elif score_type == "structure":
            prompt = """
As a virtual evaluator with expertise in English composition, your role is to critically analyze and grade student essays according to a predetermined set of rubrics in Structure dimension. You are to act as an impartial judge and evaluate the essays based on the quality of the writing and adherence to the essay prompt.

Here are the specific guidelines for the Structure score:
Structure Dimension (4 points in total)
- Level 1: 3-4 points:
    - Clearly and logically structured
    - Smooth and coherent transitions.
- Level 2: 1-2 points:
    - Mostly clearly and logically structured
    - Relatively smooth and coherent transitions
- Level 3: 0-1 point:
    - Not clearly and logically structured
    - Fragmented and disconnected structures and sentences.

Sample Essay Prompt: 
{}

Student's Essay to Evaluate
{}

Please output the Structure score only. 
""".format(student_prompt, data["essay_content"][i])
            answer = str(data["structure"][i])

        template = {
            "messages": [
                {
                    "role": "system", 
                    "content": "As a virtual evaluator with expertise in English composition, your role is to critically analyze and grade student essays according to a predetermined set of rubrics."
                }, 
                {
                    "role": "user",
                    "content": prompt
                },
                {
                    "role": "assistant",
                    "content": answer
                }
            ]
        }
        template = json.dumps(template)
        template_list.append(template)

    with open("./exam_{}_{}_json.jsonl".format(score_type, train), "w") as f:
        for temp in template_list:
            json.loads(temp)
            f.write(temp + "\n")




def try_finetuned_model(train="test"):
    data = pd.read_csv("./exam_{}.csv".format(train), index_col=0)
    print(data.head())

    response_list = []
    for i in range(len(data)):
        print(i)

        prompt = """
As a virtual evaluator with expertise in English composition, your role is to critically analyze and grade student essays according to a predetermined set of rubrics. You are to act as an impartial judge and evaluate the essays based on the quality of the writing and adherence to the essay prompt.

The total score should be assigned, where higher scores represent higher quality: 
1. Content (on a scale of 0 to 8)
2. Language (on a scale of 0 to 8)
3. Structure (on a scale of 0 to 4)
4. Total Score (on a scale of 0 to 20) = Content Score + Language Score + Structure Score

Here are the specific guidelines for each score:
{}

Sample Essay Prompt: 
{}

Student's Essay to Evaluate
{}

Please present your evaluation in the following list manner:
[Content Score, Language Score, Structure Score, Total Score]
For example: [6, 5, 3, 12]
""".format(true_rubrics, student_prompt, data["essay_content"][i])
        
        flag = 0
        while flag == 0:
            try:
                response = openai.ChatCompletion.create(
                    model="ft:gpt-3.5-turbo-1106:personal::8XCCHF4K", # overall output
                    # model="ft:gpt-3.5-turbo-1106:personal::8XM39gK8", # total score
                    # model="ft:gpt-3.5-turbo-1106:personal::8bOAZyON", # content score
                    # model="ft:gpt-3.5-turbo-1106:personal::8bOCyI5f", # language score
                    # model="ft:gpt-3.5-turbo-1106:personal::8bO2Z6y7", # structure score
                    temperature=0.0,
                    messages=[
                            {"role": "system", "content": "As a virtual evaluator with expertise in English composition, your role is to critically analyze and grade student essays according to a predetermined set of rubrics."},
                            {"role": "user", "content": prompt}
                        ]
                )

                response_content = response["choices"][0]["message"]["content"]
                flag = 1
            except:
                time.sleep(60)
                flag = 0
                print("error")
        
        response_list.append(response_content)

    data["ft_output"] = response_list
    print(data.head())
    data.to_csv("./exam_structure_{}_result.csv".format(train))




judge_examples = """
Example 1 (under-estimate):
Model: Total Score: 12, Content Score: 4.5, Language Score: 4.5, Structure Score: 3
Ground Truth: Total Score: 16, Content Score: 6, Language Score: 6, Structure Score: 4
Student Essay: 
Dear Jim, How is everything? Our school is collecting the ideas of graduation ceremony. I want to join it. And I'm writing to you for some advice. I already have some ideas about the graduation ceremony, all the First,I think we ca n let graduators write letters for we each other. Then, can sing a song together. What's more, in the end of the ceremony, we should take a photo to freeze the wonderful moment. I designed these three activities because I believe in the ceremony, all of us want to l eave some momeries. Writing letters and taking photos are the best way to realize it. And singing is also the best gift for our graduation. Do you have any other good ideas ? Expevting to your reply. Yours, LiHua

Example 2 (over-estimate):
Model: Total Score: 9, Content Score: 3.5, Language Score: 3, Structure Score: 2.5
Ground Truth: Total Score: 6, Content Score: 2, Language Score: 2, Structure Score: 2
Student Essay: 
How was it going Our school will hold a graduation ceremony. Now, I'm writing to ask your advise. This is my design First, all graduates will take photos in the playground, listening the president's speak. Then, each graduates parents will give the present to their child. Last but not least, walking in the school's. The reason why I design these activities is don't forget the memorie in our high school life. The more important is the parent's paid.We should appreciate theirs paid. I'm looking forward to hearing from you and your unexpected ideal

Example 3 (accurate):
Model: Total Score: 13, Content Score: 5, Language Score: 5, Structure Score: 3
Ground Truth: Total Score: 13, Content Score: 5, Language Score: 4.5, Structure Score: 3.5
Student Essay: 
Geetings! Our school is collecting activity plans for Grade 12 graduation ceremony. I'd like to paticipate and put forward some ideas. Hope you can give me your advice. Here are my thoughts for the ceremony. First, a PPT will be arranged to present photos of schoolmates taken in the past 3 years, and soft background music will also be played with the PPT. After that, our headmaster can deliver a speech to graduates, including wishes towards future. I design it for several reasons. I hope everyone will notice their progress and harvest in high school through my PPT.Besides, headmaster's speech is bound to bring us happiness and hope for the future. What do you think of my plan? Looking forward to your early reply.

Example 4 (accurate):
Model: Total Score: 12, Content Score: 4.5, Language Score: 4.5, Structure Score: 3
Ground Truth: Total Score: 12, Content Score: 4.5, Language Score: 4.5, Structure Score: 3
Student Essay: 
Dear Jim, How is everything go? As my school is looking for plans for the Graduation activity of Grade 12, I have some ideas about it and I am willing to join it. Knowing your school had hold this activity a few days ago, I'm writing to ask for some suggestions. In my opinion, the Graduation activity should hold in the school hall.First, schoolmaster give us a small speech about the resiposible that we will face in the future and how to deal with them. Then, our parents show up to give us a surprising gift.After that, we can have free time to take photos of school. I have a few reasons for above these.Firstly, the small speech can be of help will in the longer term, in society. Secondly, the surprising gift be a sign of being an face adult, which will encourge us to the unknown future.Lastly, the photos we took will be memorable in the later decades. May you give me some suggestion about these ideas? Looking forward to your reply. Yours, Li Hua.
"""


def get_gpt_judge_result(train="test"):
    data = pd.read_csv("./exam_{}.csv".format(train), index_col=0)
    print(data.head())

    res_df_1 = pd.read_csv("./exam_ensemble_{}_result.csv".format(train), index_col=0)
    res_df_2 = pd.read_csv("./exam_fewshot_rubrics_raw.csv", index_col=0)

    print(len(data))
    print(len(res_df_1))
    print(len(res_df_2))

    response_list = []
    for i in range(len(data)):
        print(i)

        output_1 = "Total Score: {}, Content Score: {}, Language Score: {}, Structure Score: {}".format(res_df_1["pred_total"][i], res_df_1["pred_content"][i], res_df_1["pred_language"][i], res_df_1["pred_structure"][i])

        prompt = """
Task: Virtual English Composition Evaluator

Objective: Evaluate student essays using an expert rating model's scores. Critically analyze these essays against a specific rubric to determine the appropriateness of the model's scores.

Scoring Criteria:
{}

Expert Rating Model Context:
- Accuracy: High accuracy, trained on extensive data.
- Limitations: Struggles with extreme cases.
- Usage: Generally, model scores are reliable. Direct use is often appropriate. Adjust scores only when you think it is necessary.

Instances of Inaccurate and Accurate Model Scoring:
{}

Materials for Evaluation:
1. Essay Prompt: {}
2. Student Essay: {}
3. Scores by the Expert Model: {}

Evaluation Format:
Provide your assessment in JSON format:
{{"Total Score": "Calculated Total Score", "Content Score": "Your score for Content", "Language Score": "Your score for Language", "Structure Score": "Your score for Structure", "Explanations": "Your Explanations"}}""".format(true_rubrics, judge_examples, student_prompt, data["essay_content"][i], output_1)

        flag = 0
        while flag == 0:
            try:
                response = openai.ChatCompletion.create(
                    # model="gpt-4",
                    model="gpt-4-1106-preview",
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    messages=[
                            {"role": "system", "content": "As a virtual English composition evaluator, your role is to critically analyze these essays against a specific rubric to determine the appropriateness of the model's scores."},
                            {"role": "user", "content": prompt}
                        ]
                )

                response_content = response["choices"][0]["message"]["content"]
                flag = 1
            except:
                time.sleep(60)
                flag = 0
                print("error")
        
        if i == 0:
            print(response_content)
        response_list.append(response_content)


    data["output"] = response_list
    print(data.head())
    
    data.to_csv("./exam_judge_{}_result.csv".format(train))



