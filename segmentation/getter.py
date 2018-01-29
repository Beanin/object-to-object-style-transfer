import json
import torchvision.models as models

from pascal_voc_loader import pascalVOCLoader
from fcn import *

def get_model(name, n_classes):
    model = _get_model_instance(name)

    if name in ['fcn32s', 'fcn16s', 'fcn8s']:
        model = model(n_classes=n_classes)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)
    else:
        model = model(n_classes=n_classes)

    return model

def _get_model_instance(name):
    try:
        return {
            'fcn32s': fcn32s,
            'fcn8s': fcn8s,
            'fcn16s': fcn16s,
        }[name]
    except:
        print('Model {} not available'.format(name))

def get_loader(name):
    """get_loader
    :param name:
    """
    return {
        'pascal': pascalVOCLoader,
    }[name]


def get_data_path(name, config_file='config.json'):
    """get_data_path
    :param name:
    :param config_file:
    """
    data = json.load(open(config_file))
    return data[name]['data_path']
