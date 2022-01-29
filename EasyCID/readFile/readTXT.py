import numpy as np
import chardet


def locate_line(lines, keyword):
    for i in range(0, len(lines)):
        fr = lines[i].find(keyword)
        if fr != -1:
            return i
    return "CAN'T FIND";


def parseBWTekFile(filepath, filename, select_ramanshift=False, xname='Raman Shift', yname="Dark Subtracted #1"):
    file_content = open(filepath).read()
    fc_splits = file_content.split('\n')
    spectrum = {}
    spectrum['name'] = filename[0:(len(filename) - 4)]
    spectrum['excition'] = float(fc_splits[locate_line(fc_splits, 'laser_wavelength')].split(';')[1])
    spectrum['integral_time'] = float(fc_splits[locate_line(fc_splits, 'intigration times')].split(';')[1])

    nTitle = locate_line(fc_splits, xname)

    Xunit = fc_splits[nTitle].split(';')
    AXI = Xunit.index(xname)
    DSI = Xunit.index(yname)
    xaxis_min = nTitle + 1 + int(fc_splits[locate_line(fc_splits, 'xaxis_min')].split(';')[1])
    xaxis_max = nTitle + 1 + int(fc_splits[locate_line(fc_splits, 'xaxis_max')].split(';')[1])

    nLen = xaxis_max - xaxis_min
    spectrum["axis"] = np.zeros((nLen,), dtype=np.float64)
    spectrum["spectrum"] = np.zeros((nLen,), dtype=np.float64)
    for i in range(xaxis_min, xaxis_max):
        fc_ss = fc_splits[i].split(';')
        spectrum["axis"][i - xaxis_min] = float(fc_ss[AXI])
        spectrum["spectrum"][i - xaxis_min] = float(fc_ss[DSI])
    if select_ramanshift:
        inds = np.logical_and(spectrum["axis"] > 160, spectrum["axis"] < 3000)
        spectrum["axis"] = spectrum["axis"][inds]
        spectrum["spectrum"] = spectrum["spectrum"][inds].astype(np.float64)
    return spectrum


def read_simple_txt(file_path):
    x = []
    y = []
    f = open(file_path, 'rb')
    r = f.read()
    charInfo = chardet.detect(r)
    f = open(file_path, 'r', encoding=(charInfo['encoding']))
    d = f.readlines()

    for line in d:
        d1 = line.replace(',', ' ').split()
        try:
            x.append(float(d1[0]))
            y.append(float(d1[1]))
        except:
            pass
    return np.array(x), np.array(y)