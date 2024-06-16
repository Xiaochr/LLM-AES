import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from transformers import get_linear_schedule_with_warmup
import random
from sklearn.metrics import cohen_kappa_score


from unsloth import FastLanguageModel
max_seq_length = 4096 # 8192
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./models/slow_module",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

FastLanguageModel.for_inference(model)

data = pd.read_csv("/root/autodl-tmp/data/exam_1_2_train.csv", index_col=0)



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




class EssayDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.label_mapping = {
            'overall_score': 41,
            'content_score': 17,
            'language_score': 17,
            'structure_score': 9
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inputs = self.tokenizer(
            [
                essay_prompt.format(
                    true_rubrics,
                    self.data["essay_prompt"][idx],
                    self.data["essay_content"][idx],
                    "",
                )
            ], return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        emb = outputs.hidden_states[-1][:, -1, :]
        
        emb.squeeze(0)
        emb = emb.detach().float().cpu().numpy()

        labels = [
            int(self.data["overall_score"][idx] * 2),
            int(self.data["content_score"][idx] * 2),
            int(self.data["language_score"][idx] * 2),
            int(self.data["structure_score"][idx] * 2)
        ]

        return torch.tensor(emb, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


class EssayScoringModel(nn.Module):
    def __init__(self, input_dim):
        super(EssayScoringModel, self).__init__()
        self.linear1 = nn.Linear(input_dim, 41)  # overall_score
        self.linear2 = nn.Linear(input_dim, 17)  # content_score
        self.linear3 = nn.Linear(input_dim, 17)  # language_score
        self.linear4 = nn.Linear(input_dim, 9)   # structure_score

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
    
        self.linear1.apply(init_weights)
        self.linear2.apply(init_weights)
        self.linear3.apply(init_weights)
        self.linear4.apply(init_weights)

    def forward(self, x):
        return (
            self.linear1(x),
            self.linear2(x),
            self.linear3(x),
            self.linear4(x)
        )

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

set_seed(42)



dataset = EssayDataset(data, tokenizer)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)


input_dim = model.config.hidden_size
scoring_model = EssayScoringModel(input_dim).to("cuda")


num_epochs = 20
print_interval = 500
total_steps = len(dataloader) * num_epochs


criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(scoring_model.parameters(), lr=1e-3, eps=1e-8, weight_decay=0.01)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=50,
    num_training_steps=total_steps
)

def evaluate_model(dataloader, model, criterion):
    model.eval()
    total_loss = 0
    all_labels = [[] for _ in range(4)]
    all_preds = [[] for _ in range(4)]

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to("cuda")
            labels = [label.to("cuda") for label in labels.transpose(0, 1)]

            outputs = model(inputs)
            loss = sum(criterion(output[:, 0], label) for output, label in zip(outputs, labels))
            total_loss += loss.item()

            for i, (output, label) in enumerate(zip(outputs, labels)):
                all_preds[i].extend(output[:, 0].argmax(dim=1).cpu().numpy())
                all_labels[i].extend(label.cpu().numpy())

    qwks = [cohen_kappa_score(all_labels[i], all_preds[i], weights="quadratic") for i in range(len(all_labels))]
    return total_loss / len(dataloader), qwks


test_data = pd.read_csv("./data/exam_1_2_test.csv", index_col=0)
test_dataset = EssayDataset(test_data, tokenizer)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

best_loss = float('inf')
best_qwk = 0

for epoch in range(num_epochs):
    scoring_model.train()
    running_loss = 0.0
    epoch_loss = 0.0
    for i, (inputs, labels) in enumerate(dataloader, 1):
        inputs = inputs.to("cuda")
        labels = [label.to("cuda") for label in labels.transpose(0, 1)]

        outputs = scoring_model(inputs)
        loss = sum(criterion(output[:, 0], label) for output, label in zip(outputs, labels))

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(scoring_model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        epoch_loss += loss.item()

        if i % print_interval == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Step {i}, Loss: {running_loss / print_interval:.4f}")
            running_loss = 0.0

    print(f"Epoch {epoch + 1} completed. Average Loss: {epoch_loss / len(dataloader):.4f}")

    test_loss, qwks = evaluate_model(test_dataloader, scoring_model, criterion)
    print(f"Epoch {epoch + 1} Test Loss: {test_loss:.4f}")
    print(f"QWK Scores: Overall: {qwks[0]:.4f}, Content: {qwks[1]:.4f}, Language: {qwks[2]:.4f}, Structure: {qwks[3]:.4f}")

    if test_loss < best_loss:
        best_loss = test_loss
        torch.save(scoring_model.state_dict(), "./models/fast_module.pth")
        print("Best model saved with test loss: {:.4f}".format(best_loss))


