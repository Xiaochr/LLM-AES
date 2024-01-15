import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split, RandomSampler, SequentialSampler
import re
from transformers import get_linear_schedule_with_warmup, BertTokenizer, BertModel, BertConfig
from torch.optim import AdamW
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import random
import numpy as np
import time
from sklearn.metrics import f1_score, cohen_kappa_score


# Example: ASAP set 3
file_name = "prompt3"

full_data = pd.read_excel("./ASAP/training_set_rel3.xlsx", index_col=0)
full_data.head()

data = full_data[full_data["essay_set"] == 3]
data.index = range(len(data))
label_dim = len(data["domain1_score"].drop_duplicates())

X = data["essay"].values
y = data["domain1_score"].values

X_train, X_other, y_train, y_other = train_test_split(X, y, test_size=0.8, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_other, y_other, test_size=0.5, random_state=42)


# hyper params
batch_size = 16
max_len = 512

epochs = 10
learning_rate = 1e-5 # 1e-4
epsilon = 1e-6 # 1e-8
warmup_steps = 0 # 1e2

seed = 42
is_frozen = False


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


def preprocessing_for_bert(data, tokenizer, max_len):
    """Perform required preprocessing steps for pretrained BERT.
    @param    data (np.array): Array of texts to be processed.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                tokens should be attended to by the model.
    """
    
    input_ids = []
    attention_masks = []

    for sent in data:
        encoded_sent = tokenizer.encode_plus(
            text=sent, #text_preprocessing(sent),  # Preprocess sentence
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            max_length=max_len,                  # Max length to truncate/pad
            pad_to_max_length=True,         # Pad sentence to max length
            #return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True      # Return attention mask
            )

        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks


class BERTDataset(Dataset):
    def __init__(self, txt_list, labels, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        print(len(labels))
        print(labels[0])
        self.labels = torch.tensor(labels)

        self.input_ids, self.attn_masks = preprocessing_for_bert(data=txt_list, tokenizer=tokenizer, max_len=max_length)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx], self.labels[idx]


bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


train_dataset = BERTDataset(
    txt_list=X_train.tolist(),
    labels=y_train.tolist(),
    tokenizer=bert_tokenizer,
    max_length=max_len
  )

train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)

val_dataset = BERTDataset(
    txt_list=X_val.tolist(),
    labels=y_val.tolist(),
    tokenizer=bert_tokenizer,
    max_length=max_len
  )

val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)

test_dataset = BERTDataset(
    txt_list=X_test.tolist(),
    labels=y_test.tolist(),
    tokenizer=bert_tokenizer,
    max_length=max_len
  )

test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size)


class BERTClassifier(nn.Module):
    """
        BERT for classification tasks
    """
    def __init__(self, freeze_bert=False, dim_in=768, dim_out=20):
        """
            @param    bert: a BertModel object
            @param    classifier: a torch.nn.Module classifier
            @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.classifier = nn.Linear(dim_in, dim_out)

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        self.classifier.apply(init_weights)

    def forward(self, input_ids, attention_mask):
        """
            Feed input to BERT and the classifier to compute logits.
            @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                            max_length)
            @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                            information with shape (batch_size, max_length)
            @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                            num_labels)
        """

        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)

        last_hidden_state_cls = outputs[0][:, 0, :]

        logits = self.classifier(last_hidden_state_cls)

        return logits


def set_seed(seed_value=42):
    """
        Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


set_seed(seed)


model = BERTClassifier(freeze_bert=is_frozen, dim_out=label_dim)
optimizer = AdamW(model.parameters(), lr=learning_rate, eps=epsilon)
loss_fn = nn.CrossEntropyLoss()

total_steps = len(train_dataloader) * epochs

scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps = warmup_steps,
                num_training_steps = total_steps
            )


device = torch.device("cuda")
model = model.to(device)


def topk_accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)  
        batch_size = target.size(0)
        _, y_pred = output.topk(k=maxk, dim=1)  
        y_pred = y_pred.t() 

        target_reshaped = target.view(1, -1).expand_as(y_pred)
        correct = (y_pred == target_reshaped)

        list_topk_accs = []
        for k in topk:
            ind_which_topk_matched_truth = correct[:k]
            flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(-1).float()
            tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(dim=0, keepdim=True)
            topk_acc = tot_correct_topk / batch_size
            list_topk_accs.append(topk_acc)
        return list_topk_accs




def train(model, train_dataloader, val_dataloader, test_dataloader, epochs=5, evaluation=True):
    print("Start training...\n")
    train_loss_list, train_acc_list, train_f1_list, train_acc5_list = [], [], [], []
    val_loss_list, val_acc_list, val_f1_list, val_acc5_list = [], [], [], []
    test_loss_list, test_acc_list, test_f1_list, test_acc5_list = [], [], [], []

    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Train Acc':^9} | {'Train Acc5':^9} | {'Train F1':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Val Acc5':^9} | {'Val F1':^12} | {'Elapsed':^9}")
        print("-"*150)

        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()

        # Reset tracking variables at the beginning of each epoch
        total_loss, total_acc, total_f1, total_acc5, batch_loss, batch_acc, batch_f1, batch_acc5, batch_counts = 0, 0, 0, 0, 0, 0, 0, 0, 0

        # Put the model into the training mode
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            batch_counts += 1
            # Load batch to GPU
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
            b_labels = b_labels.type(torch.LongTensor)
            b_labels = b_labels.to(device)

            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass. This will return logits.
            logits = model(b_input_ids, b_attn_mask)

            # Compute loss and accumulate the loss values
            loss = loss_fn(logits, b_labels)
            batch_loss += loss.item()
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1).flatten()

            f1 = f1_score(b_labels.cpu().data, preds.cpu().data, average="weighted")
            batch_f1 += f1
            total_f1 += f1

            topk_acc = topk_accuracy(logits, b_labels, topk=(1, 3))
            accuracy = topk_acc[0].item()
            acc5 = topk_acc[1].item()
            batch_acc += accuracy
            total_acc += accuracy
            batch_acc5 += acc5
            total_acc5 += acc5

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()

            # Print the loss values and time elapsed for every 20 batches
            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch

                # Print training results
                print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {batch_acc / batch_counts:^9.6f} | {batch_acc5 / batch_counts:^9.6f} | {batch_f1 / batch_counts:^12.6f} | {'-':^10} | {'-':^10} | {'-':^10} | {'-':^10} | {time_elapsed:^9.2f}")

                # Reset batch tracking variables
                batch_loss, batch_acc, batch_acc5, batch_f1, batch_counts = 0, 0, 0, 0, 0
                t0_batch = time.time()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)
        avg_train_acc = total_acc / len(train_dataloader)
        avg_train_acc5 = total_acc5 / len(train_dataloader)
        avg_train_f1 = total_f1 / len(train_dataloader)
        train_loss_list.append(avg_train_loss)
        train_acc_list.append(avg_train_acc)
        train_acc5_list.append(avg_train_acc5)
        train_f1_list.append(avg_train_f1)

        print("-"*150)

        output_dir = './model_save/' + file_name + "_" + str(epoch_i) + ".pt"

        print("Saving model to %s" % output_dir)
        torch.save(model, output_dir)


        # =======================================
        #               Evaluation
        # =======================================
        val_loss, val_accuracy, val_acc5, val_f1 = evaluate(model, val_dataloader)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_accuracy)
        val_acc5_list.append(val_acc5)
        val_f1_list.append(val_f1)

        # Print performance over the entire training data
        time_elapsed = time.time() - t0_epoch

        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Train Acc':^9} | {'Train Acc5':^9} | {'Train F1':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Val Acc5':^9} | {'Val F1':^12} | {'Elapsed':^9}")
        print("-"*150)
        print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {avg_train_acc:^9.6f} | {avg_train_acc5:^9.6f} | {avg_train_f1:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.6f} | {val_acc5:^9.6f} | {val_f1:^10.6f} | {time_elapsed:^9.2f}")
        print("-"*150)

        # =======================================
        #               Test
        # =======================================
        if evaluation == True:
            # After the completion of each training epoch, measure the model's performance
            # on our validation set.
            test_loss, test_accuracy, test_acc5, test_f1 = evaluate(model, test_dataloader)
            test_loss_list.append(test_loss)
            test_acc_list.append(test_accuracy)
            test_acc5_list.append(test_acc5)
            test_f1_list.append(test_f1)

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch

            print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Train Acc':^9} | {'Train Acc5':^9} | {'Train F1':^12} | {'Test Loss':^10} | {'Test Acc':^9} | {'Test Acc5':^9} | {'Test F1':^12} | {'Elapsed':^9}")
            print("-"*150)
            print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {avg_train_acc:^9.6f} | {avg_train_acc5:^9.6f} | {avg_train_f1:^12.6f} | {test_loss:^10.6f} | {test_accuracy:^9.6f} | {test_acc5:^9.6f} | {test_f1:^10.6f} | {time_elapsed:^9.2f}")
            print("-"*150)

        print("\n")

    metric_df = pd.DataFrame(
        np.array([train_loss_list, train_acc_list, train_acc5_list, train_f1_list, val_loss_list, val_acc_list, val_acc5_list, val_f1_list, test_loss_list, test_acc_list, test_acc5_list, test_f1_list]).T,
        columns=["train_loss", "train_acc", "train_acc5", "train_f1", "val_loss", "val_acc", "val_acc5", "val_f1", "test_loss", "test_acc", "test_acc5", "test_f1"])

    print(metric_df)
    metric_df.to_csv("./results/metrics_" + file_name + ".csv", encoding="utf_8_sig")
    print("Training complete!")
    return metric_df

# from sklearn.metrics import f1_score, cohen_kappa_score

def evaluate(model, val_dataloader):
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    # Tracking variables
    total_acc = 0
    total_acc5 = 0
    total_loss = 0
    total_f1 = 0

    full_b_labels, full_preds = [], []

    # For each batch in our validation set...
    for batch in val_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
        b_labels = b_labels.type(torch.LongTensor)
        b_labels = b_labels.to(device)

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)

        # Compute loss
        loss = loss_fn(logits, b_labels)
        #val_loss.append(loss.item())
        total_loss += loss.item()

        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()

        f1 = f1_score(b_labels.cpu().data, preds.cpu().data, average="weighted")
        total_f1 += f1

        full_b_labels.extend(b_labels.cpu().data)
        full_preds.extend(preds.cpu().data)

        topk_acc = topk_accuracy(logits, b_labels, topk=(1, 3))
        accuracy = topk_acc[0].item()
        acc5 = topk_acc[1].item()
        total_acc += accuracy
        total_acc5 += acc5

    # Compute the average accuracy and loss over the validation set.
    val_loss = total_loss / len(val_dataloader)
    val_acc = total_acc / len(val_dataloader)
    val_acc5 = total_acc5 / len(val_dataloader)
    val_f1 = total_f1 / len(val_dataloader)

    print("QWK: ", cohen_kappa_score(full_b_labels, full_preds, weights="quadratic"))

    return val_loss, val_acc, val_acc5, val_f1


metric_df = train(model, train_dataloader, val_dataloader, test_dataloader, epochs=epochs, evaluation=True)





