import numpy as np
import random
import time
from tqdm import tqdm
import matplotlib.pyplot as plt


def noise(spectrum, nr):
    N_MAX = nr * np.max(spectrum)
    Xnoise = np.random.normal(0, N_MAX, (1, spectrum.shape[0]))
    spectrum_noise = spectrum + Xnoise
    return np.maximum(spectrum_noise, 0)


def comb_num(m, n):
    a = 1
    b = 1
    result = 1
    minNI = min(n, m - n)
    for j in range(0, minNI):
        a = a * (m - j)
        b = b * (minNI - j)
        result = a // b
    return result


def component_in(spectra_raw, component, num, nr):
    c1 = np.random.uniform(0.1, 1, (num, 1))
    c2 = np.random.rand(num, 1)
    k = np.random.randint(0, spectra_raw.shape[0], size=(num, 1))
    Spectrumdata_new = np.zeros((1, spectra_raw.shape[1]))
    component_in = (spectra_raw[component, :]).reshape((1, spectra_raw.shape[1]))
    for i in tqdm(range(num)):
        Spectrumdata_new2 = c1[i] * component_in + c2[i] * spectra_raw[k[i], :]
        Spectrumdata_new2 = noise(Spectrumdata_new2, nr)
        Spectrumdata_new = np.vstack((Spectrumdata_new, Spectrumdata_new2))
    Spectrumdata_new = np.delete(Spectrumdata_new, 0, 0)
    label = np.ones((num, 1))
    return {'spectrum_in': Spectrumdata_new, 'label_in': label}


def component_out(spectra_raw, component, num, nr):
    c1 = np.random.rand(num, 1)
    c2 = np.random.rand(num, 1)
    k = np.random.randint(0, spectra_raw.shape[0], size=(num, 1))
    for j in range(num):
        while k[j] == component:
            k[j] = np.random.randint(0, spectra_raw.shape[0])
    h = np.random.randint(0, spectra_raw.shape[0], size=(num, 1))
    for l in range(num):
        while h[l] == component:
            h[l] = np.random.randint(0, spectra_raw.shape[0])
    Spectrumdata_new = np.zeros((1, spectra_raw.shape[1]))
    for i in range(num):
        Spectrumdata_new2 = c1[i] * spectra_raw[k[i]] + c2[i] * spectra_raw[h[i]]
        Spectrumdata_new2 = noise(Spectrumdata_new2, nr)
        Spectrumdata_new = np.vstack((Spectrumdata_new, Spectrumdata_new2))
    Spectrumdata_new = np.delete(Spectrumdata_new, 0, 0)
    label = np.zeros((num, 1))
    return {'spectrum_out': Spectrumdata_new, 'label_out': label}


def component_in_2(spectra_raw, component, num, nr, max_num_com):
    ratios = []
    new_sepctra = np.zeros((1, spectra_raw.shape[1]))
    for i in range(max_num_com - 1):
        comb_i = comb_num(spectra_raw.shape[0] - 1, i + 1)
        ratios.append(comb_i)
    ratios = np.array(ratios) / sum(ratios)
    nums = np.round(ratios * num).astype(np.int64)
    print(nums)
    for i in range(max_num_com - 1):
        spectra_i = new_component_in(spectra_raw, component, nums[i], nr, i + 2)
        new_sepctra = np.vstack((new_sepctra, spectra_i))
    new_sepctra = np.delete(new_sepctra, 0, 0)
    label = np.ones((np.sum(nums), 1))
    print(new_sepctra.shape)
    return {'spectrum_in': new_sepctra, 'label_in': label}


def component_out_2(spectra_raw, component, num, nr, max_num_com):
    ratios = []
    new_sepctra = np.zeros((1, spectra_raw.shape[1]))
    for i in range(max_num_com - 1):
        comb_i = comb_num(spectra_raw.shape[0] - 1, i + 2)
        ratios.append(comb_i)
    ratios = np.array(ratios) / sum(ratios)
    nums = np.round(ratios * num).astype(np.int64)
    print(nums)
    for i in range(max_num_com - 1):
        spectra_i = new_component_out(spectra_raw, component, nums[i], nr, i + 2)
        new_sepctra = np.vstack((new_sepctra, spectra_i))
    new_sepctra = np.delete(new_sepctra, 0, 0)
    label = np.zeros((np.sum(nums), 1))
    print(new_sepctra.shape)
    return {'spectrum_out': new_sepctra, 'label_out': label}


def new_component_in(spectra_raw, component, num, nr, max_num_com):
    a = np.zeros((num, spectra_raw.shape[0]))
    com_idxs = list(np.arange(0, spectra_raw.shape[0]))
    com_idxs.remove(component)  # remove target component
    for i in range(num):
        idx = random.sample(list(com_idxs), max_num_com - 1)
        ratios = np.random.rand(max_num_com - 1)
        for id in range(len(idx)):
            a[i, idx[id]] = ratios[id]
        a[i, component] = np.random.uniform(0.1, 1)
    new_sepctra = np.dot(a, spectra_raw)
    for i in range(num):
        new_sepctra[i] = noise(new_sepctra[i], nr)
    return new_sepctra


def new_component_out(spectra_raw, component, num, nr, max_num_com):
    a = np.zeros((num, spectra_raw.shape[0]))
    com_idxs = list(np.arange(0, spectra_raw.shape[0]))
    com_idxs.remove(component)  # remove target component
    for i in range(num):
        idx = random.sample(list(com_idxs), max_num_com)
        ratios = np.random.rand(max_num_com)
        for id in range(len(idx)):
            a[i, idx[id]] = ratios[id]
    new_sepctra = np.dot(a, spectra_raw)
    for i in range(num):
        new_sepctra[i] = noise(new_sepctra[i], nr)
    return new_sepctra


def data_augment(spectra_raw, component, num, nr=0, max_num_com=3):
    data_in = component_in_2(spectra_raw, component, num=int(num / 2), nr=nr, max_num_com=max_num_com)
    Xin = data_in['spectrum_in']
    Yin = data_in['label_in']
    data_out = component_out_2(spectra_raw, component, num=int(num / 2), nr=nr, max_num_com=max_num_com)
    Xout = data_out['spectrum_out']
    Yout = data_out['label_out']
    spectra = np.concatenate((Xin, Xout), axis=0)
    labels = np.concatenate((Yin, Yout), axis=0)
    return spectra, labels