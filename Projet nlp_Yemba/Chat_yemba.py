import streamlit as st
import torch
from tokenizers import Tokenizer
import spacy
#import torch
from Chatbot_essaie import Seq2Seq, Encoder, Decoder
from Chatbot_essaie import en_vocab, ye_vocab
from Chatbot_essaie import model
ye_nlp= Tokenizer.from_file("yemba_tokenizer.json")
en_nlp = spacy.load("en_core_web_sm")


# Détection du device

#from model import Seq2Seq, Encoder, Decoder
#import torch
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

#model = Seq2Seq(encoder, decoder, device).to(device)
model=model
# Charger le modèle sauvegardé
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#model = torch.load("seq2seq_translation_model.pth", map_location=device)
#model_path="seq2seq_translation_model.pth"
#model.load_state_dict(torch.load(model_path, map_location=device))
#model.eval()


#model.load_state_dict(torch.load("seq2seq_translation_model_bon.pth", map_location=device))
model.eval()

# entrainement du modele 


# Charger les poids du modèle sauvegardé
#model.load_state_dict(torch.load("seq2seq_translation_model.pth", map_location=device))
#model.eval()

#model.load_state_dict(torch.load("seq2seq_translation_model.pth", map_location=device))
#model.eval()  # Mode évaluation

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

# Exemple de test


# Charger le modèle
#model.load_state_dict(torch.load("seq2seq_translation_model.pth", map_location="cpu"))
#model.eval()
lower= True
unk_token = "<unk>"
pad_token = "<pad>"
sos_token = "<sos>"
eos_token = "<eos>"


# Fonction de traduction
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
sentence = st.text_input("Entrez votre texte en anglais :", "", key="input_sentence")

# Bouton pour traduire la phrase
if st.button("Traduire", key="translate_button"):
    if sentence:
        translation = translate_text(sentence)  # Remplace par ta fonction de traduction
        st.success(f"**Traduction en yemba :** {translation}")
    else:
        st.warning("Veuillez entrer une phrase en anglais.")


