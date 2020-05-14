from fastai import *
from fastai.text import *


learn = language_model_learner(data, AWD_LSTM, drop_mult=0.5, pretrained=False).to_fp32()