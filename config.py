import os

home = os.path.expanduser("~")
root_dirs = {
    'bird': home + '/Hello/Data/bird',
    'car':  home + '/Hello/Data/car',
    'air':  home + '/Hello/Data/aircraft',
    'dog':  home + '/Hello/Data/dog'
}

class_nums = {
    'bird': 200,
    'car': 196,
    'air': 100,
    'dog': 120
}

HyperParams = {
    'alpha': 0.5,
    'beta':  0.5,
    'gamma': 1,
    'kind': 'bird',
    'bs': 20,
    'epoch': 200,
    'arch': 'resnet50'
}
