def parse_boolean(value):
    value = value.lower()
    if value in ["true", "yes", "y", "1", "t"]:
        return True
    elif value in ["false", "no", "n", "0", "f"]:
        return False
    return False


def parse_plm_layer(layer):
    if layer.isdigit():
        return int(layer)
    elif layer in ["none", "emb"]:
        return layer
    else:
        return None
        
        
def str_or_none(s):
    return None if s.lower() == "none" else s


def int_or_none(x):
    return None if x.lower() == "none" else int(x)


def float_or_int(x):
    return float(x) if "." in x else int(x)