import os

SRC: os.PathLike = os.path.dirname(os.path.realpath(__file__))
BASE: os.PathLike = os.path.join(SRC, "../")
DATA: os.PathLike = os.path.join(BASE, "data/")
