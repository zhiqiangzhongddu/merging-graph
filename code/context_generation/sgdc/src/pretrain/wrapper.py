# meta learning algorithm
from . import pretrain_ours
from . import linear_evaluation
from . import pretrain_ssl

def get_algorithm(name):
    return globals()[name]
