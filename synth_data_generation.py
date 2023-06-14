import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from ydata_synthetic.synthesizers import ModelParameters
from ydata_synthetic.synthesizers.timeseries import TimeGAN
from common_utils import load_data

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

def slice_sequences(data: np.array, seq_len: int):
    """
    Функция для создания коллекции отрезков исходного временного ряда,
        на которой можно было бы обучить TimeGAN
    Параметры:
        data (np.array): значения временного ряда,
        seq_len (int) - длина каждого отрезка
    Возвращает:
        sequence_data: лист массивов длиной seq_len
        scaler - обученный объект MinMaxScaler; возвращается с целью применения
            обратной трансформации к созданным синтетическим данным, чтобы
            синтетические ряды имели тот же диапазон значений, что и исходный ряд
    """

    ori_data = data.copy()
    scaler = MinMaxScaler().fit(ori_data)
    ori_data = scaler.transform(ori_data)
    
    sequence_data = []
    for i in range(0, len(ori_data) - seq_len):
        _x = ori_data[i:i + seq_len]
        sequence_data.append(_x)
        
    return sequence_data, scaler

"""Определение параметров TimeGAN"""
SEQ_LEN = 35
N_SEQ = 1
HIDDEN_DIM = 35
GAMMA = 1

NOISE_DIM = 32
DIM = 128
BATCH_SIZE = 128

LOG_STEP = 100
LEARNING_RATE = 1e-3
TRAIN_STEPS = 10000

gan_args = ModelParameters(batch_size=BATCH_SIZE,
                           lr=LEARNING_RATE,
                           noise_dim=NOISE_DIM,
                           layers_dim=DIM)


df = load_data(f'data/series2.csv')
df = df.set_index(['ds']).sort_index()
seq_data, scaler = slice_sequences(df.values, SEQ_LEN)

"""Обучение генеративной сети"""
synth = TimeGAN(model_parameters=gan_args, 
                hidden_dim=HIDDEN_DIM, 
                seq_len=SEQ_LEN, 
                n_seq=N_SEQ, 
                gamma=GAMMA)
synth.train(seq_data, train_steps = TRAIN_STEPS)

"""Создание синтетического датасета"""
synth_data = synth.sample(50000)


"""Визуализация синтетических данных"""
plt.figure(figsize = (10, 6))

plt.subplot(211)
sns.lineplot(x = np.arange(len(synth_data[0])), y = synth_data[0])
plt.xlabel('time index', fontsize = 12)
plt.ylabel('synth series value', fontsize = 12)

plt.subplot(212)
sns.lineplot(x = np.arange(len(synth_data[1])), y = synth_data[1])
plt.xlabel('time index', fontsize = 12)
plt.ylabel('synth series value', fontsize = 12)

plt.show()


"""Визуализация TSNE"""
sample_size = 250
idx = np.random.permutation(len(seq_data))[:sample_size]

real_sample = np.asarray(seq_data)[idx]
synth_sample = synth.sample(len(seq_data))
synth_sample = np.asarray(synth_sample)[idx]

synth_data_reduced = np.asarray(synth_sample).reshape(-1, SEQ_LEN)
real_data_reduced = real_sample.reshape(-1, SEQ_LEN)

n_components = 2
pca = PCA(n_components=n_components)
tsne = TSNE(n_components=n_components, n_iter=300)
pca.fit(real_data_reduced)

pca_real = pd.DataFrame(pca.transform(real_data_reduced))
pca_synth = pd.DataFrame(pca.transform(synth_data_reduced))
data_reduced = np.concatenate((real_data_reduced, synth_data_reduced), axis=0)
tsne_results = pd.DataFrame(tsne.fit_transform(data_reduced))


fig = plt.figure(constrained_layout=True, figsize=(20,10))
spec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)

#TSNE scatter plot
ax = fig.add_subplot(spec[0,0])
ax.set_title('Результаты PCA',
             fontsize=20,
             color='red',
             pad=10)

#PCA scatter plot
plt.scatter(pca_real.iloc[:, 0].values, pca_real.iloc[:,1].values,
            c='black', alpha=0.2, label='Реальные данные')
plt.scatter(pca_synth.iloc[:,0], pca_synth.iloc[:,1],
            c='red', alpha=0.2, label='Синтетические данные')
ax.legend()

ax2 = fig.add_subplot(spec[0,1])
ax2.set_title('Результаты TSNE',
              fontsize=20,
              color='red',
              pad=10)

plt.scatter(tsne_results.iloc[:sample_size, 0].values, tsne_results.iloc[:sample_size,1].values,
            c='black', alpha=0.2, label='Original')
plt.scatter(tsne_results.iloc[sample_size:,0], tsne_results.iloc[sample_size:,1],
            c='red', alpha=0.2, label='Synthetic')

ax2.legend()



"""Сохранение результатов"""
synth_data = synth_data.reshape(synth_data.shape[0], synth_data.shape[1]).T()
synth_data = scaler.inverse_transform(synth_data)
synth_data_frame = pd.DataFrame(synth_data, 
                                columns = ['id' + str(n) for n in range(synth_data.shape[1])])
synth_data_frame.to_csv('data/synth_data.csv', sep = ';', index = False)