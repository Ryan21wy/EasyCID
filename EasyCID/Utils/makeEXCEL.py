import pandas as pd


def make_excel(mixtures, components, save_path, ratios=None):
    data = []
    if ratios:
        for i in range(len(mixtures)):
            mix = mixtures[i]
            for j in range(len(components[mix])):
                if j == 0:
                    data.append([mix, components[mix][j], ratios[i][j]])
                else:
                    data.append([' ', components[mix][j], ratios[i][j]])
        df = pd.DataFrame(data, columns=['Mixture', 'Component', 'Ratio'])
    else:
        for i in range(len(mixtures)):
            mix = mixtures[i]
            for j in range(len(components[mix])):
                if j == 0:
                    data.append([mix, components[mix][j]])
                else:
                    data.append([' ', components[mix][j]])
        df = pd.DataFrame(data, columns=['Mixture', 'Component'])
    df.to_excel(save_path, index=False)