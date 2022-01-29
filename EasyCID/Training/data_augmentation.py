import numpy as np


def noise(spectrum, nr):
    N_MAX = nr * np.max(spectrum)
    Xnoise = np.random.normal(0, N_MAX, (1, spectrum.shape[1]))
    spectrum_noise = spectrum + Xnoise
    return np.maximum(spectrum_noise, 0)


def component_in(spectra_raw, component, num, nr):
    c1 = np.random.uniform(0.1, 1, (num, 1))
    c2 = np.random.rand(num, 1)
    k = np.random.randint(0, spectra_raw.shape[0], size=(num, 1))
    Spectrumdata_new = np.zeros((1, spectra_raw.shape[1]))
    component_in = (spectra_raw[component, :]).reshape((1, spectra_raw.shape[1]))
    for i in range(num):
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


def data_augment(spectra_raw, component, num, nr=0):
    data_in = component_in(spectra_raw, component, num=int(num / 2), nr=nr)
    Xin = data_in['spectrum_in']
    Yin = data_in['label_in']
    data_out = component_out(spectra_raw, component, num=int(num / 2), nr=nr)
    Xout = data_out['spectrum_out']
    Yout = data_out['label_out']
    spectra = np.concatenate((Xin, Xout), axis=0)
    labels = np.concatenate((Yin, Yout), axis=0)
    return spectra, labels
