import numpy as np
from tqdm import tqdm
import scipy.io.wavfile
import re
from matplotlib import pyplot as plt
import os
from python_speech_features import mfcc
import json
favorite_music_path = './favorite'
favorite_musics_name = os.listdir(favorite_music_path)
fav_list = []
for name in tqdm(favorite_musics_name):
    try:
        #print(name)
        file_path = os.path.join(favorite_music_path, name)
        sample_rate, signal_all = scipy.io.wavfile.read(file_path)
        music_name_root = re.sub('.wav', '', name)
        for k in range(len(signal_all)//(sample_rate*30)):
            signal = signal_all[k*sample_rate*30:(k+1)*sample_rate*30]
            feature_mfcc = mfcc(signal, samplerate=sample_rate, nfft=2048, numcep=128, nfilt=128, winstep=0.03, winlen=0.03).astype(np.float32)
            music_name = music_name_root + '_%d.npy'%k
            fav_list.append(music_name)
            np.save('./favor_data/'+music_name, feature_mfcc)
    except:
        print("Error in %s" %name)
        print("Skip!")
json.dump(fav_list, open('fav_list.json', 'w'))

all_music_path = './togather'
all_musics_name = os.listdir(all_music_path)
all_list = []
for name in tqdm(all_musics_name):
    try:
        #print(name)
        file_path = os.path.join(all_music_path, name)
        sample_rate, signal_all = scipy.io.wavfile.read(file_path)
        music_name_root = re.sub('.wav', '', name)
        for k in range(len(signal_all) // (sample_rate * 30)):
            signal = signal_all[k * sample_rate * 30:(k + 1) * sample_rate * 30]
            feature_mfcc = mfcc(signal, samplerate=sample_rate, nfft=2048, numcep=128, nfilt=128, winstep=0.03,
                                winlen=0.03).astype(np.float32)
            print(feature_mfcc.max())
            music_name = music_name_root + '_%d.npy' % k
            all_list.append(music_name)
            np.save('./general_data/' + music_name, feature_mfcc)
    except:
        print("Error in %s" %name)
        print("Skip!")
json.dump(all_list, open('all_list.json', 'w'))


