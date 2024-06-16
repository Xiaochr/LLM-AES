from unsloth import FastLanguageModel
import pandas as pd 

max_seq_length = 4096 # 8192 # 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "./models/slow_module", # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)


data = pd.read_csv("./data/exam_1_2_test.csv", index_col=0)


true_rubrics = """
**Content Dimension (8 points in total)**
- Level 1: 6-8 points:
    - Content is complete with appropriate details
    - Expression is closely related to the topic
- Level 2: 3-5 points:
    - Content is mostly complete
    - Expression is fundamentally related to the topic
- Level 3: 0-2 points:
    - Content is incomplete
    - Expression is barely related or completely unrelated to the topic

**Language Dimension (8 points in total)**
- Level 1: 6-8 points:
    - Language is accurate with diverse sentence structures and little or no errors (2 errors or fewer, 8 points; 3-4 errors, 7 points; 5-6 errors, 6 points)
    - Language expression is mostly appropriate
- Level 2: 3-5 points:
    - Language is not quite accurate, with some variation in sentence structures and several errors, but they don't impede understanding (7-8 errors, 5 points; 9-10 errors, 4 points; 11-12 errors, 3 points)
    - Language expression is somewhat inappropriate
- Level 3: 0-2 points:
    - Language is hopelessly inaccurate with numerous language errors, hindering understanding (more than 12 errors)
    - Language expression is completely inappropriate

**Structure Dimension (4 points in total)**
- Level 1: 3-4 points:
    - Clearly and logically structured
    - Smooth and coherent transitions.
- Level 2: 1-2 points:
    - Mostly clearly and logically structured
    - Relatively smooth and coherent transitions
- Level 3: 0-1 point:
    - Not clearly and logically structured
    - Fragmented and disconnected structures and sentences.

**Grading Rules**
- When grading, focus first on the content dimension. If the essay's content is unrelated to the topic, the content score is 0, and both the language and structure scores are also 0. After determining the content score, then focus on the language dimension; generally, the language dimension level will not be higher than the content dimension level. Finally, evaluate the structure score, and then sum the three scores to obtain the overall score.
- For scoring the three dimensions, after initially determining the level, it is generally recommended to first assign the midpoint of the level (for example, if the language dimension is determined to be Level 1, initially assign 7 points). Then, adjust the score by adding or subtracting points to obtain the final specific score.
"""



essay_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are an experienced English teacher analyzing high school students' essays according to a specific rubric. Evaluate the following essay based on three dimensions: Content, Language, and Structure, and provide the overall assessment. 

Please provide your evaluation in the following JSON format:
{{
	"content": {{
		"completeness": "Check whether the content is complete, does the essay cover all required points? Provide your explanations with examples in the essay.",
		"topic_relevance": "Does the essay closely related to the given topic? If the content is irrelevant to the topic, then assign score 0.",
		"content_details": "Are the details and expression sufficient? Provide your explanations with examples in the essay.",
		"score_level": "Determine content dimension level based on the analyses above.",
		"score_point": "Determine content score based on the analyses above and the score level."
	}}, 
	"language": {{
		"error_details": "List all the grammar or spelling errors in the essay.",
		"error_cnt": "Total number of errors.",
		"accuracy_and_diversity": "Check the language accuracy, appropriateness, and diversity. Provide your explanations with examples in the essay.",
		"score_level": "Determine language dimension level based on the analyses above.",
		"score_point": "Determine language score based on the analyses above and the score level."
	}}, 
	"structure": {{
		"clarity": "Check whether the structure is clear and logical. Provide your explanations with examples in the essay.",
		"coherence": "Check for smooth and coherent transitions. Provide your explanations with examples in the essay.",
		"score_level": "Determine language dimension level based on the analyses above.",
		"score_point": "Determine language score based on the analyses above and the score level."
	}}, 
	"overall": {{
		"overall_assessment": "The overall assessment of the essay.",
		"score_point": "Determine the overall score."
	}}
}}

### Input:
Scoring rubric:
{}

Essay Prompt:
{}

Student's Essay to Evaluate:
{}

### Response:
{}"""






FastLanguageModel.for_inference(model) # Enable native 2x faster inference

text_list = []
for i in range(len(data)):
    print(i)
    inputs = tokenizer(
    [
        essay_prompt.format(
            true_rubrics,
            data["essay_prompt"][i], 
            data["essay_content"][i],
            "", # output - leave this blank for a generation!
        )
    ], return_tensors = "pt").to("cuda")

    outputs = model.generate(**inputs, max_new_tokens = 2048, use_cache = True)
    temp_text = tokenizer.batch_decode(outputs)[0]
    text_list.append(temp_text)
    if i % 50 == 0:
        print(temp_text)

    
data["raw_output"] = text_list

data.to_csv("./results/slow_module_results.csv")










