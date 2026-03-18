from tdc.single_pred import Tox

data = Tox(name='Carcinogens_Lagunin')

split = data.get_split(method='scaffold')
print(len(split['train']), len(split['valid']), len(split['test']))