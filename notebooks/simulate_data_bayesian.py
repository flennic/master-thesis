import sys

import numpy as np

import utilities as utils
from utilities import *

import gp_utilities as gp_utils
from gp_utilities import *

# Parameters, do not change!
input_gdansk = "../data/age_decades/"
output_dir = "../data/preprocessed/"

data_source_type = "gdansk"
splice_type = "constant"

splits = [0.6, 0.2, 0.2]
seed = 42
N = 48 * 2
samples = 100 # Easy to take quite lot in the beginning, can always be easily filtered later on

np.random.seed(seed=seed)

try:
    stop_early = int(sys.argv[1])
except:
    stop_early=None

print(f"stop_early is set to: {stop_early}")

recordings = utils.read_gdansk_to_dict(input_gdansk)
cleared_recordings = gp_utils.remove_outliers_in_recordings(recordings, in_seconds=True)
train_recordings, val_recordings, test_recordings = np.array_split(cleared_recordings, (np.array(splits)[:-1].cumsum() * len(cleared_recordings)).astype(int))

train_recordings = gp_utils.c_splice_lod_constant_by_number(train_recordings, n=N)
val_recordings = gp_utils.c_splice_lod_constant_by_number(val_recordings, n=N)
test_recordings = gp_utils.c_splice_lod_constant_by_number(test_recordings, n=N)

del recordings, cleared_recordings

print("\n\nSimulating testing data.\n")
gp_utils.simulate_data(test_recordings, set_selector="test", n=samples, stop_early=stop_early)
print("\nSimulating validation data.\n")
gp_utils.simulate_data(val_recordings, set_selector="val", n=samples, stop_early=stop_early)
print("\nSimulating training data.\n")
gp_utils.simulate_data(train_recordings, set_selector="train", n=samples, stop_early=stop_early)


