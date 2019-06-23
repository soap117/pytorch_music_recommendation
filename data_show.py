import json
import numpy as np
import os
from tqdm import tqdm
from model import RNNembModel
import torch
# generate embeddings of tsv form which can be visualized in http://projector.tensorflow.org/
save_path = './save256'
data_path = './general_data'
emb_model = RNNembModel(128, 128, 256)
emb_model.cuda()
cache_emb = os.path.join(save_path, 'Embeder_%d.pkl' % (90000))
with open(cache_emb, "rb") as f:
    params = torch.load(f)
    emb_model.load_state_dict(params)
emb_model.eval()
name_list = os.listdir(data_path)
embeddings = np.load(cache_emb)
TSV_file = open('music_emb.tsv', 'w')
for name in tqdm(name_list):
    mfcc_feature = torch.Tensor(np.load(os.path.join(data_path, name))).cuda().view(1, 1000, 128)
    with torch.no_grad():
        emb = emb_model(mfcc_feature)[0].detach().cpu().numpy()
    temp = ''
    for i in range(len(emb)):
        temp += str(emb[i])
        if i < len(emb)-1:
            temp += '\t'
        else:
            temp += '\n'
    TSV_file.writelines(temp)
TSV_file.close()
TSV_file_name = open('music_emb_name.tsv', 'w', encoding='utf-8')
for name in name_list:
    TSV_file_name.writelines(name+'\n')
TSV_file_name.close()