import os
import requests
from EasyCID.Utils.makeDir import mkdir

components = ['1,2-Dichloroethane', 'Acetonitrile', 'cyclohexane', 'Dichloromethane',
              'Diethylene glycol dimethyl ether', 'ethanol', 'isopropanol', 'methanol',
              'n-Butanol', 'n-Heptane', 'n-Heptanol', 'n-Nonane', 'Toluene']
mixtures = ['FiveSolvent-1', 'FiveSolvent-2', 'FourSolvent-1', 'FourSolvent-2', 'ThreeSolvent-1', 'ThreeSolvent-2']
db = ['EasyCID_Demo']
character_trans = {',': '%2C', ' ': '%20'}


def nameTrans(name):
    trans_list = list(character_trans.keys())
    for t in trans_list:
        name = name.replace(t, character_trans[t])
    return name


def download(url, filename):
    r = requests.get(url)
    with open(filename, "wb") as f:
        f.write(r.content)


def demo():
    p_url = "https://raw.githubusercontent.com/Ryan21wy/EasyCID/master/Samples"
    abs_dir = os.path.dirname(os.path.abspath(__file__))
    t_dir = os.path.dirname(abs_dir)
    components_path = os.path.join(t_dir, 'demo', 'components')
    mkdir(components_path)
    for com in components:
        component_file_name = com + '.txt'
        component_url = p_url + '/components/' + nameTrans(component_file_name)
        component_path = os.path.join(components_path, component_file_name)
        download(component_url, component_path)

    mixtures_path = os.path.join(t_dir, 'demo', 'mixtures')
    mkdir(mixtures_path)
    for mix in mixtures:
        mixture_file_name = mix + '.txt'
        mixture_url = p_url + '/mixtures/' + nameTrans(mixture_file_name)
        mixture_path = os.path.join(mixtures_path, mixture_file_name)
        download(mixture_url, mixture_path)

    models_path = os.path.join(t_dir, 'demo', 'models')
    mkdir(models_path)
    for com in components:
        model_file_name = com + '.h5'
        model_url = p_url + '/models/' + nameTrans(model_file_name)
        model_path = os.path.join(models_path, model_file_name)
        download(model_url, model_path)

    db_path = os.path.join(t_dir, 'demo', 'EasyCID_demo.db')
    db_url = p_url + '/EasyCID_demo.db'
    download(db_url, db_path)