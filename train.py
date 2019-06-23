from model import FullModel, RNNembModel
from sample import music_data
import torch
import numpy as np
from tqdm import tqdm
import os
db = music_data('./general_data')
save_path = './save256'
model = FullModel(128, 128, 256)
emb_model = RNNembModel(128, 128, 256)
emb_model.train()
emb_model.cuda()
model.train()
model.cuda()
opt_model = torch.optim.Adam(model.parameters(), lr=1e-4)
opt_emb = torch.optim.Adam(emb_model.parameters(), lr=1e-4)
for i in tqdm(range(100000), mininterval=1):
    opt_emb.zero_grad()
    opt_model.zero_grad()
    batch_index = np.random.randint(0, len(db.file_list), 32)
    batch_mfcc = db.get_batch(batch_index)
    batch_emb = emb_model(batch_mfcc)
    loss = model(batch_mfcc, batch_emb)
    opt_model.step()
    opt_emb.step()
    if i % 100==0:
        print(loss)
        sample_index = [0, 1, 100, 101, 200, 201]
        sample_mfcc = db.get_batch(sample_index)
        sample_emb = emb_model(sample_mfcc)
        print(sample_emb)
    if i % 10000 == 0 and i > 0:
        cache_file = os.path.join(save_path, 'Listener_%d.pkl' % (i))
        print("saving model to {}".format(cache_file))
        with open(cache_file, "wb") as f:
            params = model.state_dict()
            torch.save(params, f)
        cache_file = os.path.join(save_path, 'Embeder_%d.pkl' % (i))
        print("saving model to {}".format(cache_file))
        with open(cache_file, "wb") as f:
            params = emb_model.state_dict()
            torch.save(params, f)

