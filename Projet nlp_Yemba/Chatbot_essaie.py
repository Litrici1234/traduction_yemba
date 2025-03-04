
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import spacy
#import datasets
import torchtext
import tqdm
import pandas as pd 
from tokenizers import Tokenizer
import streamlit as st
import torch
from tokenizers import Tokenizer
import spacy
from datasets import Dataset, DatasetDict


seed = 1234

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True




def load_dataset_from_txt(file_path):
    """Charge un fichier texte et retourne une liste de dictionnaires {"lang1": ..., "lang2": ...}"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')  # Suppose que les colonnes sont séparées par une tabulation
            if len(parts) == 2:
                data.append({"ye": parts[0], "en": parts[1]})
    return Dataset.from_list(data)

def load_dataset_from_txt2(file_path):
    """Charge un fichier texte et retourne une liste de dictionnaires {"lang1": ..., "lang2": ...}"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(';')  # Suppose que les colonnes sont séparées par une tabulation
            if len(parts) == 2:
                data.append({"ye": parts[0], "en": parts[1]})
    return Dataset.from_list(data)

# Charger les fichiers
train_data = load_dataset_from_txt2("train2.txt")
validation_data = load_dataset_from_txt("validation.txt")
test_data = load_dataset_from_txt("test.txt")

# Construire le DatasetDict
dataset = DatasetDict({
    "train": train_data,
    "validation": validation_data,
    "test": test_data
})

ye_nlp= Tokenizer.from_file("yemba_tokenizer.json")
en_nlp = spacy.load("en_core_web_sm")


def tokenize_example(example, en_nlp, ye_tokenizer, max_length, lower, sos_token, eos_token):
    # Tokenisation en anglais avec spaCy (en_nlp)
    en_tokens = [token.text for token in en_nlp.tokenizer(example["en"])][:max_length]
    
    # Tokenisation en Yemba avec mon tokenizer personnalisé (BPE)
    ye_tokens = ye_tokenizer.encode(example["ye"]).tokens[:max_length]
    
    # Option de mise en minuscules
    if lower:
        en_tokens = [token.lower() for token in en_tokens]
        ye_tokens = [token.lower() for token in ye_tokens]
    
    # Ajouter les tokens de début (sos_token) et de fin (eos_token)
    en_tokens = [sos_token] + en_tokens + [eos_token]
    ye_tokens = [sos_token] + ye_tokens + [eos_token]
    
    return {"en_tokens": en_tokens, "ye_tokens": ye_tokens}


max_length = 1000
lower = True
sos_token = "<sos>"
eos_token = "<eos>"

fn_kwargs = {
    "en_nlp": en_nlp,
    "ye_tokenizer": ye_nlp,
    "max_length": max_length,
    "lower": lower,
    "sos_token": sos_token,
    "eos_token": eos_token,
}

train_data = train_data.map(tokenize_example, fn_kwargs=fn_kwargs)
valid_data = validation_data.map(tokenize_example, fn_kwargs=fn_kwargs)
test_data = test_data.map(tokenize_example, fn_kwargs=fn_kwargs)

## vocabulaire

min_freq = 2
unk_token = "<unk>"
pad_token = "<pad>"
sos_token = "<sos>"
eos_token = "<eos>"

special_tokens = [
    unk_token,
    pad_token,
    sos_token,
    eos_token,
]

en_vocab = torchtext.vocab.build_vocab_from_iterator(
    train_data["en_tokens"],
    min_freq=min_freq,
    specials=special_tokens,
)

ye_vocab = torchtext.vocab.build_vocab_from_iterator(
    train_data["ye_tokens"],
    min_freq=min_freq,
    specials=special_tokens,
)



assert en_vocab[unk_token] == ye_vocab[unk_token]
assert en_vocab[pad_token] == ye_vocab[pad_token]

unk_index = en_vocab[unk_token]
pad_index = en_vocab[pad_token]

ye_vocab.set_default_index(unk_index)
en_vocab.set_default_index(unk_index)

## numérisation du vocabulaire

def numericalize_example(example, ye_vocab, en_vocab):
    ye_ids = ye_vocab.lookup_indices(example["ye_tokens"])
    en_ids = en_vocab.lookup_indices(example["en_tokens"])
    return {"ye_ids": ye_ids, "en_ids": en_ids}

fn_kwargs = {"ye_vocab": ye_vocab, "en_vocab": en_vocab}

train_data = train_data.map(numericalize_example, fn_kwargs=fn_kwargs)
valid_data = valid_data.map(numericalize_example, fn_kwargs=fn_kwargs)
test_data = test_data.map(numericalize_example, fn_kwargs=fn_kwargs)


data_type = "torch"
format_columns = ["en_ids", "ye_ids"]

train_data = train_data.with_format(
    type=data_type, columns=format_columns, output_all_columns=True
)

valid_data = valid_data.with_format(
    type=data_type,
    columns=format_columns,
    output_all_columns=True,
)

test_data = test_data.with_format(
    type=data_type,
    columns=format_columns,
    output_all_columns=True,
)


def get_collate_fn(pad_index):
    def collate_fn(batch):
        batch_ye_ids = [example["ye_ids"] for example in batch]
        batch_en_ids = [example["en_ids"] for example in batch]
        batch_ye_ids = nn.utils.rnn.pad_sequence(batch_ye_ids, padding_value=pad_index)
        batch_en_ids = nn.utils.rnn.pad_sequence(batch_en_ids, padding_value=pad_index)
        batch = {
            "ye_ids": batch_ye_ids,
            "en_ids": batch_en_ids,
        }
        return batch

    return collate_fn

def get_data_loader(dataset, batch_size, pad_index, shuffle=False):
    collate_fn = get_collate_fn(pad_index)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
    )
    return data_loader

batch_size = 10

train_data_loader = get_data_loader(train_data, batch_size, pad_index, shuffle=True)
valid_data_loader = get_data_loader(valid_data, batch_size, pad_index)
test_data_loader = get_data_loader(test_data, batch_size, pad_index)


# Vérification de la structure
#print(dataset)


class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src length, batch size]
        src_embedding = self.embedding(src)
        embedded = self.dropout(src_embedding)
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs = [src length, batch size, hidden dim * n directions]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # outputs are always from the top hidden layer
        return hidden, cell
    
class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hidden dim]
        # context = [n layers, batch size, hidden dim]
        input = input.unsqueeze(0)
        # input = [1, batch size]
        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch size, embedding dim]
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # output = [seq length, batch size, hidden dim * n directions]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # seq length and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, hidden dim]
        # hidden = [n layers, batch size, hidden dim]
        # cell = [n layers, batch size, hidden dim]
        prediction = self.fc_out(output.squeeze(0))
        # prediction = [batch size, output dim]
        return prediction, hidden, cell
    

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert (
            encoder.hidden_dim == decoder.hidden_dim
        ), "Hidden dimensions of encoder and decoder must be equal!"
        assert (
            encoder.n_layers == decoder.n_layers
        ), "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio):
        # src = [src length, batch size]
        # trg = [trg length, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        batch_size = trg.shape[1]
        trg_length = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        # tensor to store decoder outputs
        outputs = torch.zeros(trg_length, batch_size, trg_vocab_size).to(self.device)
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # first input to the decoder is the <sos> tokens
        input = trg[0, :]
        # input = [batch size]
        for t in range(1, trg_length):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)
            # output = [batch size, output dim]
            # hidden = [n layers, batch size, hidden dim]
            # cell = [n layers, batch size, hidden dim]
            # place predictions in a tensor holding predictions for each token
            outputs[t] = output
            # decide if we are going to use teacher forcing or not
            teacher_force = False
            # get the highest predicted token from our predictions
            top1 = output.argmax(1)
            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1
            # input = [batch size]
        return outputs
    

## entrainement du modéle 

def get_collate_fn(pad_index):
    def collate_fn(batch):
        batch_ye_ids = [example["ye_ids"] for example in batch]
        batch_en_ids = [example["en_ids"] for example in batch]
        batch_ye_ids = nn.utils.rnn.pad_sequence(batch_ye_ids, padding_value=pad_index)
        batch_en_ids = nn.utils.rnn.pad_sequence(batch_en_ids, padding_value=pad_index)
        batch = {
            "ye_ids": batch_ye_ids,
            "en_ids": batch_en_ids,
        }
        return batch

    return collate_fn

def get_data_loader(dataset, batch_size, pad_index, shuffle=False):
    collate_fn = get_collate_fn(pad_index)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
    )
    return data_loader

batch_size = 10

train_data_loader = get_data_loader(train_data, batch_size, pad_index, shuffle=True)
valid_data_loader = get_data_loader(valid_data, batch_size, pad_index)
test_data_loader = get_data_loader(test_data, batch_size, pad_index)


input_dim = len(en_vocab)
output_dim = len(ye_vocab)
encoder_embedding_dim = 256
decoder_embedding_dim = 256
hidden_dim = 512
n_layers = 2
encoder_dropout = 0.5
decoder_dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = Encoder(
    input_dim,
    encoder_embedding_dim,
    hidden_dim,
    n_layers,
    encoder_dropout,
)

decoder = Decoder(
    output_dim,
    decoder_embedding_dim,
    hidden_dim,
    n_layers,
    decoder_dropout,
)

model = Seq2Seq(encoder, decoder, device).to(device)


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


model.apply(init_weights)

# Charger le modèle sauvegardé

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.load_state_dict(torch.load("seq2seq_translation_model.pth", map_location=device))
#model.eval()

#model = torch.load("seq2seq_translation_model.pth", map_location=device)
#model_path="seq2seq_translation_model.pth"
#model.load_state_dict(torch.load(model_path, map_location=device))
#model.eval()
#model.eval()
# contruction du modele
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=pad_index)


def train_fn(
    model, data_loader, optimizer, criterion, clip, teacher_forcing_ratio, device
):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(data_loader):
        src = batch["en_ids"].to(device)
        trg = batch["ye_ids"].to(device)
        # src = [src length, batch size]
        # trg = [trg length, batch size]
        optimizer.zero_grad()
        output = model(src,trg, teacher_forcing_ratio)
        # output = [trg length, batch size, trg vocab size]
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        # output = [(trg length - 1) * batch size, trg vocab size]
        trg = trg[1:].view(-1)
        # trg = [(trg length - 1) * batch size]
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        
    return epoch_loss / len(data_loader)

def evaluate_fn(model, data_loader, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            src = batch["en_ids"].to(device)
            trg = batch["ye_ids"].to(device)
            # src = [src length, batch size]
            # trg = [trg length, batch size]
            output = model(src, trg, 0)  # turn off teacher forcing
            # output = [trg length, batch size, trg vocab size]
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            # output = [(trg length - 1) * batch size, trg vocab size]
            trg = trg[1:].view(-1)
            # trg = [(trg length - 1) * batch size]
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(data_loader)

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=pad_index)

n_epochs = 20
clip = 1.0
teacher_forcing_ratio = 0.5

best_valid_loss = float("inf")

for epoch in tqdm.tqdm(range(n_epochs)):
    train_loss = train_fn(
        model,
        train_data_loader,
        optimizer,
        criterion,
        clip,
        teacher_forcing_ratio,
        device,
    )
    valid_loss = evaluate_fn(
        model,
        valid_data_loader,
        criterion,
        device,
    )
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), "tut1-model.pt")
    print(f"\tTrain Loss: {train_loss:7.3f} | Train PPL: {np.exp(train_loss):7.3f}")
    print(f"\tValid Loss: {valid_loss:7.3f} | Valid PPL: {np.exp(valid_loss):7.3f}")

def translate_sentence(
    sentence,
    model,
    ye_nlp,
    en_nlp,
    ye_vocab,
    en_vocab,
    lower,
    sos_token,
    eos_token,
    device,
    max_output_length=25,
):
    model.eval()
    with torch.no_grad():
        # Si l'entrée est une chaîne de caractères, on la tokenise avec spacy
        if isinstance(sentence, str):
            doc = en_nlp(sentence)  # Traitement de la phrase en utilisant spacy
            tokens = [token.text for token in doc][:max_output_length]  # Extraire les tokens
        else:
            tokens = [token for token in sentence]  # Si c'est déjà une liste de tokens
        
        # Mise en minuscule si demandé
        if lower:
            tokens = [token.lower() for token in tokens]
        
        # Ajouter les tokens SOS et EOS
        tokens = [sos_token] + tokens + [eos_token]

        # Convertir les tokens en indices en utilisant le vocabulaire anglais
        ids = en_vocab.lookup_indices(tokens)  # Utilisation du vocabulaire anglais
        tensor = torch.LongTensor(ids).unsqueeze(1).to(device)  # Ajouter une dimension pour le batch
        
        # Encodage de la phrase anglaise
        hidden, cell = model.encoder(tensor)
        
        # Initialisation du décodeur avec le token SOS yemba
        inputs = [ye_vocab[sos_token]]
        translated_tokens = []

        # Décodage jusqu'à la longueur maximale de sortie
        for _ in range(max_output_length):
            inputs_tensor = torch.LongTensor([inputs[-1]]).to(device)  # Dernier token du décodeur
            output, hidden, cell = model.decoder(inputs_tensor, hidden, cell)  # Décodage de la phrase
            
            predicted_token = output.argmax(-1).item()  # Prendre le token prédit
            inputs.append(predicted_token)
            
            # Si le token EOS est généré, on arrête la traduction
            if predicted_token == ye_vocab[eos_token]:
                break
            translated_tokens.append(predicted_token)

        # Convertir les indices en tokens yemba
        translated_tokens = [ye_vocab.lookup_token(idx) for idx in translated_tokens]

    return translated_tokens  # Retourner les tokens traduits


#Fonction de traduction
def translate_text(sentence):
    tokens= translate_sentence(
    sentence, 
    model,
    ye_nlp, 
    en_nlp, 
    ye_vocab, 
    en_vocab, 
    lower,
    sos_token,
    eos_token,
    device,)
    #tokens = translate_sentence(sentence, model, ye_nlp, en_nlp, ye_vocab, en_vocab, True, "<sos>", "<eos>", "cpu")
    return " ".join(tokens)

# Interface utilisateur avec Streamlit
st.title("Chatbot de Traduction Anglais → Yemba")
st.write("Entrez une phrase en anglais et obtenez la traduction en yemba.")

# Saisie utilisateur
sentence = st.text_input("Entrez votre texte en anglais :", "")

if st.button("Traduire"):
    if sentence:
        translation = translate_text(sentence)
        st.success(f"**Traduction en yemba :** {translation}")
    else:
        st.warning("Veuillez entrer une phrase en anglais.")




