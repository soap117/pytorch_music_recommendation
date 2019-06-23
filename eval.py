import numpy as np
import scipy.io.wavfile
import re
import sys
import torch
import os
from model import RNNembModel
from python_speech_features import mfcc
if __name__ == "__main__":
    wav_file = sys.argv[1]
    threshold = float(sys.argv[2])
    save_path = './save256'
    emb_model = RNNembModel(128, 128, 256)
    emb_model.cuda()
    centers = np.load('centers.npy')
    cache_emb = os.path.join(save_path, 'Embeder_%d.pkl' % (90000))
    with open(cache_emb, "rb") as f:
        params = torch.load(f)
        emb_model.load_state_dict(params)
    emb_model.eval()

    embeddings = []
    sample_rate, signal_all = scipy.io.wavfile.read(wav_file)
    if signal_all.shape[1] > 1:
        #merge if conatins more than one channel
        signal_all = np.mean(signal_all, 1)
    for k in range(len(signal_all)//(sample_rate*30)):
        #take mean result of all 30s pieces
        signal = signal_all[k*sample_rate*30:(k+1)*sample_rate*30]
        feature_mfcc = mfcc(signal, samplerate=sample_rate, nfft=2048, numcep=128, nfilt=128, winstep=0.03, winlen=0.03).astype(np.float32)
        feature_mfcc = torch.Tensor(feature_mfcc).cuda().view(1, 1000, 128)
        with torch.no_grad():
            emb = emb_model(feature_mfcc)[0].detach().cpu().numpy()
            embeddings.append(emb)
    embeddings = np.array(embeddings)
    mean_emb = np.mean(embeddings, 0)
    min_dis = 9999
    id = 0
    for k in range(len(centers)):
        dis = np.mean(np.square(centers[k] - mean_emb))
        if dis < min_dis:
            min_dis = dis
            id = k
    if min_dis > threshold:
        print("Probably don't like it")
        print("Min dis: %f" %min_dis)
    else:
        print("Probably like it")
        print("Min dis: %f" % min_dis)
        print("Closest with center %d" %id)