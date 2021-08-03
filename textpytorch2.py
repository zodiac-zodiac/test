import torch
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
from torch.utils.data import DataLoader
from torch import nn
import time
from torchtext.utils import download_from_url, unicode_csv_reader
from torchtext.data.datasets_utils import _RawTextIterableDataset
from torchtext.data.datasets_utils import _wrap_split_argument
from torchtext.data.datasets_utils import _add_docstring_header
import os
import io
import csv
import logging
from torchtext.utils import download_from_url, extract_archive, unicode_csv_reader
from torchtext.data.utils import ngrams_iterator
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from tqdm import tqdm
import sys


###############################################################################
#review https://pytorch.org/text/_modules/torchtext/datasets/text_classification.html setupdatases

def ITER_DATA(split):
    path = "C:\\Users\\nader.ammari\\Desktop\\Nuance\\SSE\\CyberAI\\cyberAI\\textpytorch-vocab\\data\\" + split + ".csv"
    file = open(path)
    reader = csv.reader(file)
    NUM_LINES= len(list(reader))
    return _RawTextIterableDataset("ITER_DATA", NUM_LINES,
                                   _create_data_from_csv(path))

def _create_data_from_csv(data_path):
        with io.open(data_path, encoding="utf8") as f:
            reader = unicode_csv_reader(f)
            for row in reader:
                yield int(row[0]), ' '.join(row[1:])



##############################################################################




##############################################################################
#input prep 



tokenizer = get_tokenizer('basic_english')
train_iter =  ITER_DATA(split='train') 

print("Usage: python cyber_AI.py [train] ")
counter = Counter()
for (label, line) in train_iter:
    counter.update(tokenizer(line))
vocab = Vocab(counter, min_freq=1)

text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]
label_pipeline = lambda x: int(x) - 1

#Generate data batch and iterator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
         label_list.append(label_pipeline(_label))
         processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
         text_list.append(processed_text)
         offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)

train_iter = ITER_DATA(split='train')

dataloader = DataLoader(train_iter, batch_size=8, shuffle=False, collate_fn=collate_batch)



#Define the model
class TextClassificationModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

#instance 

train_iter = ITER_DATA(split='train')
num_class = len(set([label for (label, text) in train_iter]))
vocab_size = len(vocab)
emsize = 64
model = TextClassificationModel(vocab_size, emsize, num_class).to(device)


##train and evaluate func 



def train(dataloader):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, (label, text, offsets) in enumerate(dataloader):

        optimizer.zero_grad()
        predited_label = model(text, offsets)
        #print(predited_label)
        #print(label)
        loss = criterion(predited_label, label)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predited_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                              total_acc/total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()

def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predited_label = model(text, offsets)
            loss = criterion(predited_label, label)
            total_acc += (predited_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc/total_count


#split data and run 


from torch.utils.data.dataset import random_split
# Hyperparameters
EPOCHS = 5 # epoch
LR = 5  # learning rate
BATCH_SIZE = 64 # batch size for training

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
total_accu = None
train_iter, test_iter = ITER_DATA(split="train"), ITER_DATA(split="test")
train_dataset = list(train_iter)
test_dataset = list(test_iter)
num_train = int(len(train_dataset) * 0.95)
split_train_, split_valid_ = random_split(train_dataset, [num_train, len(train_dataset) - num_train])

train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=True, collate_fn=collate_batch)

if len(sys.argv) > 1:

    if sys.argv[1] == "train" :
        for epoch in range(1, EPOCHS + 1):
            epoch_start_time = time.time()
            train(train_dataloader)
            accu_val = evaluate(valid_dataloader)
            if total_accu is not None and total_accu > accu_val:
              scheduler.step()
            else:
               total_accu = accu_val
            print('-' * 59)
            print('| end of epoch {:3d} | time: {:5.2f}s | '
                  'valid accuracy {:8.3f} '.format(epoch,
                                                   time.time() - epoch_start_time,
                                                   accu_val))
            print('-' * 59)
            #save !!!!!!
            torch.save(model.state_dict(), 'saved_weights.pt')


### load saved model

path='C:\\Users\\nader.ammari\\Desktop\\Nuance\\SSE\\CyberAI\\cyberAI\\textpytorch-vocab\\saved_weights.pt'
model.load_state_dict(torch.load(path));
model.eval();


### predict 
###################save

ag_news_label = {
8: "False positive",
1: "By Design",
2: "By Network",
3: "Cleansing Func",
4: "By OS",
5: "Report to lib mantainer",
6: "Accept Risk",
7: "Comment + Cleansing" }

def predict(text, text_pipeline):
    with torch.no_grad():
        text = torch.tensor(text_pipeline(text))
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item() + 1

ex_text_str = "117 Technique : M2 :  Establish and maintain control over all of your outputs \n Specifics : Added below regex in logback configuration file: <encoder> <Pattern>%d %p [%t] %logger %replace(%m){'\r?\n', '\\r\\n'}%n</Pattern> </encoder> Remaining \n Risk : None Verification : Manual log inspection."

model = model.to("cpu")

print("Based on the TSRV, this should be approved if the mitigation type submitted is < %s > " %ag_news_label[predict(ex_text_str, text_pipeline)])




