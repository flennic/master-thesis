import os
import math
import copy
import subprocess
import collections
import numpy as np
import pandas as pd
import hrvanalysis as hrv
import multiprocessing as mp
from datetime import datetime


## Gdansk
def read_gdansk_to_dict(data_dir, in_seconds=False):
    recordings = []
    for i, filename in enumerate(os.listdir(data_dir)):

        print(str(round(i/len(os.listdir(data_dir))*100, 2)) + "%", end="\r")

        recordings.append(gdansk_to_dict(data_dir + filename, False, in_seconds))
    return recordings


def gdansk_to_dict(filepath, raw_pythonic=True, in_seconds=False):
    
    recording = {}
    
    denom = 1000 if in_seconds else 1
    
    with open(filepath) as raw_recording:
        
        recording["Gender"] = filepath.split("/")[-1][0:1]
        recording["AgeDecade"] = filepath.split("/")[-1][1:3]
        recording["RecordingStartTime"] = datetime.strptime(filepath.split("/")[-1][4:9], '%H.%M').time()
        
        series = {"ContractionNo": [], "ContractionNoNorm": [], "RrInterval": []}
        
        first_index = None
        previous_ContractionNo = None
        
        for line in raw_recording:
            
            # Handling shifted indexes
            if first_index is None:
                first_index = int(line.split()[1])
            
            # Fill missing data with None's
            if previous_ContractionNo is not None:
                diff = abs(previous_ContractionNo - int(line.split()[1]))
                
                if diff > 1:
                    
                    filling_indexes = np.array(range(previous_ContractionNo+1, int(line.split()[1])))
                    
                    series["ContractionNo"].extend(filling_indexes)
                    series["ContractionNoNorm"].extend(filling_indexes - first_index)
                    series["RrInterval"].extend([math.nan]*(diff-1))
          
            series["ContractionNo"].append(int(line.split()[1]))
            series["ContractionNoNorm"].append(int(line.split()[1]) - first_index)
            series["RrInterval"].append(int(line.split()[0])/denom)
            
            previous_ContractionNo = int(line.split()[1])
            
        if raw_pythonic:
            recording["Recording"] = series
            recording["RecordingStartTime"] = str(recording["RecordingStartTime"])
        else:
            recording["Recording"] = pd.DataFrame(series)
            
        return recording


## PhysioNet
def extract_header_information_generator(path):
    with open(path, "r") as file:
        
        content = file.read()
        start, end = 0, 0
        
        while content.find(">", end) != -1:

            start = content.find(">", end)
            end = content.find("<", start)
            value = content[start+2:end-2].strip()
            
            if value[-1] == ",":
                value = value[0:-1]
            
            if value.find("#") != -1:
                value = value.split("#")[0]
            
            yield value


def read_physionet_to_dict(data_dir, sub_dirs, in_seconds=False):
    all_recordings = []

    denom = 1000 if in_seconds else 1

    for medicationdir in sub_dirs:

        print(f"Parsing {medicationdir} directory...")

        basedir = data_dir + medicationdir
        recordings = list(set([filename.split(".")[0] for filename in os.listdir(basedir) if len(filename.split(".")) == 2 and filename.find("index") == -1]))

        for i, recording in enumerate(recordings):

            # Status
            print(str(round(i/len(recordings)*100, 2)) + "%", end="\r")

            entry = {}

            # Filename
            entry["Name"] = recording[1:4]
            entry["Medication"] = recording[0]
            entry["Treatment"] = recording[4] == "b"

            # Header
            header_information = [value for value in extract_header_information_generator(basedir + recording + ".hea")]
            entry["AgeDecade"] = header_information[0]
            entry["Gender"] = header_information[1]

            try:
                entry["RrLow"] = int(float(header_information[3]) * 1000)
            except ValueError:
                entry["RrLow"] = None

            try:
                entry["RrHigh"] = int(float(header_information[4]) * 1000)
            except ValueError:
                entry["RrHigh"] = None

            # Recording
            rrintervals = subprocess.check_output(["ann2rr", "-r", recording, "-a", "atr", "-i", "s"], cwd=basedir).splitlines()
            rrintervals = np.array([int(float(value)*1000) for value in rrintervals], dtype=object)
            rrintervals = rrintervals[1:]/denom

            # This outlier detection should be handled somewhere else. Uncomment to enable.
            #try:
            #    rrintervals[(rrintervals > entry["RrHigh"]) | (rrintervals < entry["RrLow"])] = np.nan
            #except TypeError:
            #    rrintervals[abs(rrintervals - np.mean(rrintervals)) < sigma_outlier * np.std(rrintervals)] = np.nan

            entry["Recording"] = pd.DataFrame({"ContractionNoNorm": list(range(len(rrintervals))), "RrInterval": rrintervals})

            all_recordings.append(entry)
            
    return all_recordings


def class_weights_from_path(path, no_classes, label='label', inverse=True, sep=',', header=True):
    
    index = 0 if type('label') is str else label
    counter = collections.Counter({i:1 for i in range(no_classes)})
    
    with open(path) as file:
        if index == 0 and header:
            header = next(file).split(sep)
            try:
                index = header.index(label)
            except ValueError:
                ValueError(f"Label '{label}' not found in header.")
    
        if index == 0:
            raise ValueError("Index not given and could not be inferred.")
        
        for i, line in enumerate(file):
            counter[int(line.split(sep)[index])] += 1

    scores = np.array([counter[key]for key in sorted(counter.keys())])
    scores = 1/scores if inverse else scores
            
    return scores / np.sum(scores)


def class_weights_from_path_(path, label='label', inverse=True):
    df = pd.read_csv(path, index_col=0)
    scores = df[label].value_counts().sort_index()
    scores = 1/scores if inverse else scores
    return np.array(scores / np.sum(scores))


def oversample_df(df, label='label'):
    oversample_count = df[label].value_counts().max()
    return df.groupby(label, group_keys=False).apply(lambda x: x.sample(oversample_count, replace=True))


def undersample_df(df, label='label'):
    undersample_count = df[label].value_counts().min()
    return df.groupby(label, group_keys=False).apply(lambda x: x.sample(undersample_count))


def clear_data_in_recordings(recordings, interpolation_method='linear', in_seconds=True):
    
    cleared_recordings = []
    recordings_deepcopy = copy.deepcopy(recordings)
    denom = 1000 if in_seconds else 1
    
    for i, recording in enumerate(recordings_deepcopy):
        
        # Status
        print(str(round(i/len(recordings_deepcopy)*100, 2)) + "%", end="\r")
        
        recording["Recording"]["RrInterval"] = hrv.preprocessing.get_nn_intervals(recording["Recording"]["RrInterval"],
                                                                    interpolation_method=interpolation_method,
                                                                    limit_area=None,
                                                                    limit_direction="both",
                                                                    verbose=False)
        recording["Recording"]["RrInterval"] = recording["Recording"]["RrInterval"] / denom
        cleared_recordings.append(recording)
        
    return cleared_recordings


def decade_to_label(decade, classification=True):
    
    radius = 5
    
    if "-" in decade:
        decade = decade[0:2]
        radius = 2.5
    
    if classification:
        return int(int(decade)/10) - 2
    return int(decade) + radius


def recording_to_x_y_feature_classification(recording):
    time_domain_features = hrv.get_time_domain_features(recording["Recording"]["RrInterval"])
    geometrical_features = hrv.get_geometrical_features(recording["Recording"]["RrInterval"])
    frequency_domain_features = hrv.get_frequency_domain_features(recording["Recording"]["RrInterval"])
    csi_cvi_features = hrv.get_csi_cvi_features(recording["Recording"]["RrInterval"])
    poincare_plot_features = hrv.get_poincare_plot_features(recording["Recording"]["RrInterval"])

    feature_dictionary = {
                            **time_domain_features,
                            **geometrical_features,
                            **frequency_domain_features,
                            **csi_cvi_features,
                            **poincare_plot_features
                         }
    
    x = [value for value in feature_dictionary.values()]
    y = decade_to_label(recording["AgeDecade"], True)
    
    return [y] + x


def recording_to_x_y_feature_regression(recording):
    time_domain_features = hrv.get_time_domain_features(recording["Recording"]["RrInterval"])
    geometrical_features = hrv.get_geometrical_features(recording["Recording"]["RrInterval"])
    frequency_domain_features = hrv.get_frequency_domain_features(recording["Recording"]["RrInterval"])
    csi_cvi_features = hrv.get_csi_cvi_features(recording["Recording"]["RrInterval"])
    poincare_plot_features = hrv.get_poincare_plot_features(recording["Recording"]["RrInterval"])

    feature_dictionary = {
                            **time_domain_features,
                            **geometrical_features,
                            **frequency_domain_features,
                            **csi_cvi_features,
                            **poincare_plot_features
                         }
    
    x = [value for value in feature_dictionary.values()]
    y = decade_to_label(recording["AgeDecade"], False)
    
    return [y] + x


def get_feature_names(recording):
    time_domain_features = hrv.get_time_domain_features(recording["Recording"]["RrInterval"])
    geometrical_features = hrv.get_geometrical_features(recording["Recording"]["RrInterval"])
    frequency_domain_features = hrv.get_frequency_domain_features(recording["Recording"]["RrInterval"])
    csi_cvi_features = hrv.get_csi_cvi_features(recording["Recording"]["RrInterval"])
    poincare_plot_features = hrv.get_poincare_plot_features(recording["Recording"]["RrInterval"])

    feature_dictionary = {
                            **time_domain_features,
                            **geometrical_features,
                            **frequency_domain_features,
                            **csi_cvi_features,
                            **poincare_plot_features
                         }
    
    return [key for key in feature_dictionary.keys()]


def recordings_to_feature_dataframe(recordings, classification=True):
    
    y_name = "label" if classification else "age"

    cores = mp.cpu_count()
    pool = mp.Pool(cores)
    if classification:
        results = pool.imap(recording_to_x_y_feature_classification, recordings)
    else:
        results = pool.imap(recording_to_x_y_feature_regression, recordings)
    pool.close()
    pool.join()
    results = list(results)
    
    column_names = get_feature_names(recordings[0])
    data_frame = pd.DataFrame(results)
    data_frame.columns = [y_name] + column_names
    data_frame[y_name] = data_frame[y_name].astype('int32')

    return data_frame


def pad(pad_list, size, padding):
    pad_list = list(pad_list)
    pad_list = pad_list[0:min(len(pad_list), size)]
    return pad_list + [padding] * abs((len(pad_list)-size))


def recording_to_deep_classification_physionet(recording):
    label = decade_to_label(recording["AgeDecade"], True)
    series = pad(recording["Recording"]["RrInterval"], pad_length_global, 0)
    return [label] + series


def recording_to_deep_regression_physionet(recording):
    label = decade_to_label(recording["AgeDecade"], False)
    series = pad(recording["Recording"]["RrInterval"], pad_length_global, 0)
    return [label] + series


def recording_to_deep_classification_gdansk(recording):
    label = decade_to_label(recording["AgeDecade"], True)
    series = pad(recording["Recording"]["RrInterval"], pad_length_global, 0)
    return [label] + series


def recording_to_deep_regression_gdansk(recording):
    label = decade_to_label(recording["AgeDecade"], False)
    series = pad(recording["Recording"]["RrInterval"], pad_length_global, 0)
    return [label] + series


def recordings_to_deep_dataframe(recordings,
                                 pad_length,
                                 classification=True,
                                 gdansk=True):
    
    global pad_length_global
    pad_length_global = pad_length
    #print(pad_length_global)

    y_name = "label" if classification else "age"
    column_names = [y_name] + ["rr" + str(i + 1) for i in range(pad_length)]
        
    cores = mp.cpu_count()
    pool = mp.Pool(cores)
    
    if classification and gdansk:
        results = pool.imap(recording_to_deep_classification_gdansk, recordings)
    elif not classification and gdansk:
        results = pool.imap(recording_to_deep_regression_gdansk, recordings)
    elif classification and not gdansk:
        results = pool.imap(recording_to_deep_classification_physionet, recordings)
    else:
        results = pool.imap(recording_to_deep_regression_physionet, recordings)
        
    pool.close()
    pool.join()
    results = list(results)
    
    data_frame = pd.DataFrame(results)
    data_frame.columns = column_names
    data_frame[y_name] = data_frame[y_name].astype('int32')

    return data_frame


def splice_series_constant(series_df, chunksize_in_minutes=5, data_is_seconds=True):
    unit_factor = 1 if data_is_seconds else 1000
    split_size = unit_factor * chunksize_in_minutes * 60 / np.mean(series_df["RrInterval"])
    n = len(series_df) // split_size
    return np.array_split(series_df, n)


def splice_series_random(series_df, n=100, chunksize_in_minutes=5, data_is_seconds=True, sigma=50):
    unit_factor = 1 if data_is_seconds else 1000
    split_size = int(unit_factor * chunksize_in_minutes * 60 / np.mean(series_df["RrInterval"]))
    cap = len(series_df) - split_size
    starts = np.random.randint(cap, size=n)
    ends = [max(min(round(np.random.normal(start+split_size, sigma)), cap+split_size), start+1) for start in starts]
    splices = [series_df.iloc[start:end,] for start, end in zip(starts, ends)]
    return splices


def splice_lod_constant_by_cunksize(lod, chunksize_in_minutes=5, data_is_seconds=True):
    spliced_recordings = []
    
    for i, recording in enumerate(lod):
        
        # Status
        print(str(round(i/len(lod)*100, 2)) + "%", end="\r")
        
        splices = splice_series_constant(recording["Recording"], chunksize_in_minutes, data_is_seconds)
        
        for splice in splices:
            recording_deepcopy = copy.deepcopy(recording)
            recording_deepcopy["Recording"] = splice
            spliced_recordings.append(recording_deepcopy)
    
    return spliced_recordings


def splice_lod_constant_by_number(lod, n=48):
    spliced_recordings = []
    
    for i, recording in enumerate(lod):
        
        # Status
        print(str(round(i/len(lod)*100, 2)) + "%", end="\r")
        
        splices = np.array_split(recording["Recording"], n)
        
        for splice in splices:
            recording_deepcopy = copy.deepcopy(recording)
            recording_deepcopy["Recording"] = splice
            spliced_recordings.append(recording_deepcopy)
    
    return spliced_recordings


def splice_lod_random(lod, n=100, chunksize_in_minutes=5, data_is_seconds=True, sigma=50):
    spliced_recordings = []
    
    for i, recording in enumerate(lod):
        
        # Status
        print(str(round(i/len(lod)*100, 2)) + "%", end="\r")
        
        splices = splice_series_random(recording["Recording"], n, chunksize_in_minutes, data_is_seconds, sigma)
        
        for splice in splices:
            recording_deepcopy = copy.deepcopy(recording)
            recording_deepcopy["Recording"] = splice
            spliced_recordings.append(recording_deepcopy)
        
    return spliced_recordings


def accuracy_score_from_label_chunks_original(real, predicted, N=48):
    real_chunks = np.array_split(real, int(real.shape[0]/N))
    pred_chunks = np.array_split(predicted, int(predicted.shape[0]/N))
    
    classification_result = []
    
    for real_chunk, pred_chunk in zip(real_chunks, pred_chunks):
        real_label = real_chunk.tolist()[0]
        majority_voted_label = np.bincount(pred_chunk).argmax()
        classification_result.append(majority_voted_label == real_label)
        
    return np.mean(classification_result)


def accuracy_score_from_label_chunks_simulated(real, predicted, N, simulated):
    
    complete_size = N * simulated
    splits = list(range(complete_size, predicted.shape[0], complete_size))
    
    # Split into whole series
    series_real = np.split(real, splits)
    series_predicted = np.split(predicted, splits)
    
    # Remove last if not yet fully simulated
    if len(series_real[-1]) != len(series_real[-2]):
        series_predicted = series_predicted[:-1]
        series_real = series_real[:-1]
        
    res = []

    for real_series_fold, pred_series_fold in zip(series_real, series_predicted):

        real_series_fold = np.array(real_series_fold)
        pred_series_fold = np.array(pred_series_fold)

        for i in range(simulated):
            indeces = np.array(range(i, complete_size, simulated))

            slice_real = real_series_fold[indeces]
            slice_pred = pred_series_fold[indeces]

            real_label = slice_real.tolist()[0]
            majority_voted_label = np.bincount(slice_pred).argmax()

            res.append(real_label == majority_voted_label)
        
    return np.mean(res)


def accuracy_score_from_label_chunks(real, predicted, N=48, simulated=None):
    
    if simulated is None:
        return accuracy_score_from_label_chunks_original(real, predicted, N)
    return accuracy_score_from_label_chunks_simulated(real, predicted, N=N, simulated=simulated)
    

def regression_results_to_label(values, no_classes=7):
    pure_labels = (np.array(values)//10 - 2).astype(int)
    labels_no_neg = np.maximum([0]*len(pure_labels), pure_labels)
    labels_no_pos = np.minimum([no_classes-1]*len(labels_no_neg), labels_no_neg)
    return labels_no_pos
