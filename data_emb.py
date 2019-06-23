import json
import numpy as np
import os
from tqdm import tqdm
from model import RNNembModel
import torch
# generate embeddings of numpy form
save_path = './save256'
emb_model = RNNembModel(128, 128, 256)
emb_model.cuda()
cache_emb = os.path.join(save_path, 'Embeder_%d.pkl' % (90000))
with open(cache_emb, "rb") as f:
    params = torch.load(f)
    emb_model.load_state_dict(params)
emb_model.eval()

data_path = './general_data'
name_list = os.listdir(data_path)
embeddings = []
for name in tqdm(name_list):
    mfcc_feature = torch.Tensor(np.load(os.path.join(data_path, name))).cuda().view(1, 1000, 128)
    with torch.no_grad():
        emb = emb_model(mfcc_feature)[0].detach().cpu().numpy()
        embeddings.append(emb)
np.save('embeddings_total.npy', np.array(embeddings))

data_path = './favor_data'
name_list = os.listdir(data_path)
embeddings = []
for name in tqdm(name_list):
    mfcc_feature = torch.Tensor(np.load(os.path.join(data_path, name))).cuda().view(1, 1000, 128)
    with torch.no_grad():
        emb = emb_model(mfcc_feature)[0].detach().cpu().numpy()
        embeddings.append(emb)
np.save('embeddings_favor.npy', np.array(embeddings))

