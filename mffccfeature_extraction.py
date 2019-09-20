import numpy as np
from speechpy.feature import mfcc
import os
import soundfile as sf
import pickle
from sklearn.svm import SVC

Datasource="/home/rabianajeeb/python_projects/Datasets/SER_Data/Training_Data"
def min_max_scalling(data):
    normalzied_list = []
    for value in data:
        normalzied_list.append((value - np.min(data)) /
                       (np.max(data)-np.min(data)))
    return np.array(normalzied_list)

def get_featurefrom_mfcc(file_path:str, mfcc_len: int):
    signal,speed=sf.read(file_path)
    signal_len=len(signal)
    speed=48000
    meansignal_length = 176578

    if signal_len < meansignal_length:
        pad_len= meansignal_length-signal_len
        pad_rem = pad_len % 2
        pad_len //= 2
        new_signal=np.pad(signal,(pad_len,pad_len+pad_rem),"constant",constant_values=0)
    else:
        pad_len=signal_len-meansignal_length
        pad_len//=2
        new_signal=signal[pad_len:pad_len+meansignal_length]
    print(new_signal)
    # print(len(new_signal))
    mel_coefficients = mfcc(new_signal, speed, num_cepstral=mfcc_len)
    mel_coefficients=np.ravel(mel_coefficients)
    normalized_feature = min_max_scalling(mel_coefficients)
    return normalized_feature

def getting_sourcedata(datasource = Datasource,mfcc_value=70,classlabels=("Angry","Happy","Neutral","Sad")):
    data=[]
    labels=[]
    names=[]
    os.chdir(datasource)
    for i,directory in enumerate(classlabels):
        os.chdir(directory)
        for filename in os.listdir('.'):
            filepath=os.getcwd()+'/'+filename
            features=get_featurefrom_mfcc(file_path=filepath,mfcc_len=mfcc_value)
            data.append(features)
            labels.append(i)
            names.append(filename)
        os.chdir("..")
    return np.array(data), np.array(labels)

if __name__ == "__main__":
     data, labels = getting_sourcedata()
     with open("/home/rabianajeeb/python_projects/Datasets/SER_Data/Training_Data/my_data.txt", "wb") as f:
          pickle.dump(data,f)
     with open("/home/rabianajeeb/python_projects/Datasets/SER_Data/Training_Data/my_label.txt", "wb") as f:
         pickle.dump(labels,f)







