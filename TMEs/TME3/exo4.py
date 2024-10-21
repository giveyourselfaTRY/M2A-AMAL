import string
import unicodedata
import torch
import sys
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset,DataLoader

from utils import RNN_trump, device

## Liste des symboles autorisés
LETTRES = string.ascii_letters + string.punctuation+string.digits+' '
## Dictionnaire index -> lettre
id2lettre = dict(zip(range(1,len(LETTRES)+1),LETTRES))
id2lettre[0]='' ##NULL CHARACTER
## Dictionnaire lettre -> index
lettre2id = dict(zip(id2lettre.values(),id2lettre.keys()))

def normalize(s):
    """ Nettoyage d'une chaîne de caractères. """
    return ''.join(c for c in unicodedata.normalize('NFD', s) if  c in LETTRES)

def string2code(s):
    """ Transformation d'une chaîne de caractère en tenseur d'indexes """
    return torch.tensor([lettre2id[c] for c in normalize(s)])

def code2string(t):
    """ Transformation d'une liste d'indexes en chaîne de caractères """
    if type(t) !=list:
        t = t.tolist()
    return ''.join(id2lettre[i] for i in t)

class TrumpDataset(Dataset):
    def __init__(self,text,maxsent=None,maxlen=None):
        """  Dataset pour les tweets de Trump
            * text : texte brut
            * maxsent : nombre maximum de phrases.
            * maxlen : longueur maximale des phrases.
        """
        maxlen = maxlen or sys.maxsize
        full_text = normalize(text)
        self.phrases = [p[:maxlen].strip()+"." for p in full_text.split(".") if len(p)>0]
        if maxsent is not None:
            self.phrases=self.phrases[:maxsent]
        self.MAX_LEN = max([len(p) for p in self.phrases])

    def __len__(self):
        return len(self.phrases)
    def __getitem__(self,i):
        t = string2code(self.phrases[i])
        t = torch.cat([torch.zeros(self.MAX_LEN-t.size(0),dtype=torch.long),t])
        return t[:-1],t[1:]



#  TODO:
PATH = "../data/"
batch_size=32

data_trump = DataLoader(TrumpDataset(open(PATH+"trump_full_speech.txt","rb").read().decode(),maxlen=1000), batch_size= batch_size, shuffle=True)

vocab_size=len(LETTRES)+1
embedding_dim=128
hidden_dim=256
output_dim=len(LETTRES)+1
epochs=10
lr=0.005

model=RNN_trump(vocab_size,embedding_dim,hidden_dim,output_dim)
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=lr)
writer = SummaryWriter(log_dir='./logs')


for epoch in range(epochs):
    model.train()
    total_loss=0.0
    for x_batch,y_batch in data_trump:
        x_batch,y_batch=x_batch.to(device),y_batch.to(device)
        hidden=model.init_hidden(x_batch.size(0)).to(device)

        optimizer.zero_grad()
        outputs,hidden=model(x_batch,hidden)
        outputs = outputs.view(-1, output_dim)  # Flatten the output
        y_batch = y_batch.view(-1)

        loss=criterion(outputs,y_batch)
        loss.backward()
        optimizer.step()

        total_loss+=loss.item()
    avg_loss=total_loss / len(data_trump)
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss}')
    writer.add_scalar('Training Loss', avg_loss, epoch)
writer.close()

def generate_sequence(start_text, max_length=1000,temperature=0.9):
    model.eval()
    generated_sequence = string2code(start_text).unsqueeze(0).to(device)  # Start with the given text
    hidden = model.init_hidden(1).to(device)  # Batch size of 1

    while generated_sequence.size(1)<max_length:
        output, hidden = model(generated_sequence[:, -1:], hidden)  # Generate one symbol at a time
        output=output[:,-1,:]/temperature
        proba=torch.softmax(output,dim=-1)
        # Get the most probable symbol
        output_symbols = torch.multinomial(proba, num_samples=1)

        # Check if the generated symbol is a number, if so, skip it
        generated_char = id2lettre[output_symbols.item()]
        if generated_char.isdigit():
            continue  # Skip appending if the character is a digit

        generated_sequence = torch.cat((generated_sequence, output_symbols), dim=1)

    return code2string(generated_sequence.squeeze(0).cpu().tolist())

# Example generation
start_text = "We are going "
print(generate_sequence(start_text))




