# read and load the dataset
import json
import pandas as pd
# word2vec model
from gensim.models import Word2Vec
import nltk
from datasets import load_dataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

with open('Psychology-10K.json') as f:
    data = json.load(f)
df = pd.DataFrame(data)
df.head()


# read and load the dataset
import json
import pandas as pd
with open('Psychology-10K.json') as f:
    data = json.load(f)
df = pd.DataFrame(data)
df.head()


# find unique count in input
print("unique elements in input:", df['input'].nunique())
# find unique count in output
print("unique elements in output:", df['output'].nunique())
# almost all are useful training data


# new datframe with only input and output
df = df[['input', 'output']]

# convert to list of dict
data = df.to_dict(orient='records')


alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

#dataset = load_dataset("yahma/alpaca-cleaned", split = "train")
dataset = load_dataset("samhog/psychology-10k", split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True,)

# test train split
from datasets import Dataset
from sklearn.model_selection import train_test_split

train, test = train_test_split(dataset["text"], test_size=0.2)
train_dataset = Dataset.from_dict({"text": train})
test_dataset = Dataset.from_dict({"text": test})

# concatenate
# SPX is the in and out seperation token
# add end of line token to the input
# NDUX is the end of line token
df['text_for_training'] = df['input'] + ' SPX ' + df['output'] + " NDUX"



nltk.download('punkt')

df['text_for_training_words'] = df['text_for_training'].apply(nltk.word_tokenize)

# train word2vec model
model_word_2vec = Word2Vec((df['text_for_training_words'].values), min_count=1)
model_word_2vec.save('word2vec.model')

# apply word2vec model
def word2vec(data):
    return model_word_2vec.wv[data]

df['vec_for_training'] = df['text_for_training_words'].apply(word2vec)

max_len = 152

def pad(data):
    return np.pad(data, ((0, max_len - len(data)), (0, 0)))

# RNN model with words as input, a word is a 100 dim vector, build in torch, use X_train as input for the text generation model

# define a n layer RNN model with LSTM, large
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out)
        return out
    
# hyperparameters
input_size = 100
hidden_size = 256
output_size = 100
n_layers = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# instantiate the model
model = RNN(input_size, hidden_size, output_size, n_layers).to(device)

# loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# train the model
num_epochs = 100
for epoch in range(num_epochs):
    for data in X_train:
        data = torch.Tensor(data).unsqueeze(0).to(device)
        output = model(data)
        loss = criterion(output, data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'epoch {epoch+1}/{num_epochs}, loss={loss.item()}')



# synthesizing text using the model
input_text = "I'm feeling really angry and I don't know how to control it. Can you help me?"
input_text = nltk.word_tokenize(input_text)
input_text = [model_word_2vec.wv[word] for word in input_text]
output_text = generate_text(input_text, model)
output_text = np.array(output_text)
output_text = [model_word_2vec.wv.index2word[word] for word in output_text]
output_text = ' '.join(output_text)
