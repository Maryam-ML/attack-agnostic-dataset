from typing import Dict
from copy import deepcopy
from dfadetect.models import lcnn 
def get_model(model_name: str, config: Dict, device:str):
    if model_name == "rawnet":
        return raw_net2.RawNet(deepcopy(RAW_NET_CONFIG), device=device)
    elif model_name == "mesonet_inception":
        return mesonet.MesoInception4(num_classes=1, **config)
    elif model_name == "lcnn":
        return lcnn.LCNN(**config)
    elif model_name == "xception":
        return xception.xception(num_classes=1, pretrained=None, **config)
    else:
        raise ValueError(f"Model '{model_name}' not supported")