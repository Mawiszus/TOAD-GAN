# %%
from typing import List
from utils import load_pkl, save_pkl
import torch
import transformers
model = transformers.BertModel.from_pretrained('bert-base-uncased')
model.eval()
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

prepath = "/home/schubert/projects/TOAD-GAN/input/minecraft/ruins/"
token_dict = load_pkl(
    "representations", prepath)
token_list = list(token_dict.keys())
token_names: List[str] = [token.replace("minecraft:", "").replace("_", " ")
                          for token in token_list]

natural_token_dict = {}
with torch.no_grad():
    for token_name, token in zip(token_names, token_list):
        ids = tokenizer.encode(token_name)
        tokens = tokenizer.convert_ids_to_tokens(ids)
        bert_output = model.forward(torch.tensor(
            ids).unsqueeze(0), encoder_hidden_states=True)
        final_layer_embeddings = bert_output[0][-1]
        natural_token_dict[token] = final_layer_embeddings[0]

save_pkl(natural_token_dict, "natural_representations", prepath)
