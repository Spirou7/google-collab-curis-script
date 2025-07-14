import tensorflow as tf
from local_tpu_resolver import LocalTPUClusterResolver

from models.resnet import resnet_18
from models.backward_resnet import backward_resnet_18
from models.resnet_nobn import resnet_18_nobn
from models.backward_resnet_nobn import backward_resnet_18_nobn
from models import efficientnet
from models import backward_efficientnet
from models import densenet
from models import backward_densenet
from models import nf_resnet
from models import backward_nf_resnet

import config
from prepare_data import generate_datasets
import math
import os
import argparse
import numpy as np
from models.inject_utils import *
from injection import read_injection
