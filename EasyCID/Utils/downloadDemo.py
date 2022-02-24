import requests


def get_url_info():
    components = ['1,2-Dichloroethane', 'Acetonitrile', 'cyclohexane', 'Dichloromethane',
                  'Diethylene glycol dimethyl ether', 'ethanol', 'isopropanol', 'methanol',
                  'n-Butanol', 'n-Heptane', 'n-Heptanol', 'n-Nonane', 'Toluene']
    mixtures = ['FiveSolvent-1', 'FiveSolvent-2', 'FourSolvent-1', 'FourSolvent-2', 'ThreeSolvent-1', 'ThreeSolvent-2']
    model_info = ['ModelsInfo.json']
    db = ['EasyCID_demo']
    return [components, mixtures, model_info, db]


def nameTrans(name):
    character_trans = {',': '%2C', ' ': '%20'}
    trans_list = list(character_trans.keys())
    for t in trans_list:
        name = name.replace(t, character_trans[t])
    return name


def download(url, filename):
    r = requests.get(url)
    with open(filename, "wb") as f:
        f.write(r.content)