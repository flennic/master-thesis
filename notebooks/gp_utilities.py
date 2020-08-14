import os

import time
import copy
import logging
import os.path
import subprocess

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import hrvanalysis as hrv

import utilities as utils
from utilities import *

from os import path
from datetime import datetime, timedelta

# TensorFlow
logging.disable(logging.WARNING)
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels


def get_distance_df(kernel, start=0, end=10, stepsize=200):
    with tf.device('/cpu:0'):
        results = []
        values = np.arange(start, end, end/stepsize)
        
        for i, value in enumerate(values):
            x = values.reshape(-1, 1)
            y = (np.zeros(len(x))+values[i]).reshape(-1, 1)
            results.append(kernel.apply(x, y).numpy())

        df = pd.DataFrame(results)
        df.columns = values
        df.index = values
        
        return df
    
    
def plot_kernel(kernel, start=0, end=10, stepsize=200, title=None, xlabel=None, ylabel=None, ax=None):
    df = get_distance_df(kernel, start=start, end=end, stepsize=stepsize)
    if ax is None:
        ax = sns.heatmap(df, cmap="viridis", xticklabels=stepsize//5, yticklabels=stepsize//5)
    else:
        sns.heatmap(df, cmap="viridis", xticklabels=stepsize//5, yticklabels=stepsize//5, ax=ax)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_ylabel(xlabel)
    ax.set_xlabel(ylabel)
    return ax


def get_kernel_bayesian(smooth_amplitude, smooth_length_scale,
                    periodic_amplitude, periodic_length_scale, periodic_period, periodic_local_amplitude, periodic_local_length_scale,
                    global_periodic_amplitude, global_periodic_length_scale, global_periodic_period,
                    irregular_amplitude, irregular_length_scale, irregular_scale_mixture,
                    matern_onehalf_amplitude, matern_onehalf_length_scale,
                    matern_threehalves_amplitude, matern_threehalves_length_scale,
                    matern_fivehalves_amplitude, matern_fivehalves_length_scale):
    
    # Smooth kernel
    smooth_kernel = tfk.ExponentiatedQuadratic(
        amplitude=smooth_amplitude,
        length_scale=smooth_length_scale)
    
    # Local periodic kernel
    local_periodic_kernel = (
        tfk.ExpSinSquared(
            amplitude=periodic_amplitude, 
            length_scale=periodic_length_scale,
            period=periodic_period) * 
        tfk.ExponentiatedQuadratic(
            amplitude=periodic_local_amplitude,
            length_scale=periodic_local_length_scale))
    
    # Periodic
    global_periodic_kernel = tfk.ExpSinSquared(
            amplitude=global_periodic_amplitude, 
            length_scale=global_periodic_length_scale,
            period=global_periodic_period)
    
    # Irregular kernel
    irregular_kernel = tfk.RationalQuadratic(
        amplitude=irregular_amplitude,
        length_scale=irregular_length_scale,
        scale_mixture_rate=irregular_scale_mixture)
    
    # Matern 1/2
    matern_onehalf_kernel = tfk.MaternOneHalf(
        amplitude = matern_onehalf_amplitude,
        length_scale = matern_onehalf_length_scale
    )
    
    # Matern 3/2
    matern_fivehalf_kernel = tfk.MaternThreeHalves(
        amplitude = matern_threehalves_amplitude,
        length_scale = matern_threehalves_length_scale
    )
    
    # Matern 5/2
    matern_fivehalf_kernel = tfk.MaternFiveHalves(
        amplitude = matern_fivehalves_amplitude,
        length_scale = matern_fivehalves_length_scale
    )
    
    return smooth_kernel + local_periodic_kernel + irregular_kernel + matern_onehalf_kernel + matern_fivehalf_kernel + matern_fivehalf_kernel


def build_gp(X_train, Y_train):
    
    def build_gp_internal(smooth_amplitude, smooth_length_scale,
              periodic_amplitude, periodic_length_scale, periodic_period, periodic_local_amplitude, periodic_local_length_scale,
              global_periodic_amplitude, global_periodic_length_scale, global_periodic_period,
              irregular_amplitude, irregular_length_scale, irregular_scale_mixture,
              matern_onehalf_amplitude, matern_onehalf_length_scale,
              matern_threehalves_amplitude, matern_threehalves_length_scale,
              matern_fivehalves_amplitude, matern_fivehalves_length_scale,
              observation_noise_variance):
        """Defines the conditional dist. of GP outputs, given kernel parameters."""
        
        # Create the covariance kernel, which will be shared between the prior (which we
        # use for maximum likelihood training) and the posterior (which we use for
        # posterior predictive sampling)
        kernel = get_kernel_bayesian(smooth_amplitude, smooth_length_scale,
                  periodic_amplitude, periodic_length_scale, periodic_period, periodic_local_amplitude, periodic_local_length_scale,
                  global_periodic_amplitude, global_periodic_length_scale, global_periodic_period,
                  irregular_amplitude, irregular_length_scale, irregular_scale_mixture,
                  matern_onehalf_amplitude, matern_onehalf_length_scale,
                  matern_threehalves_amplitude, matern_threehalves_length_scale,
                  matern_fivehalves_amplitude, matern_fivehalves_length_scale)
        
        # Create the GP prior distribution, which we will use to train the model
        # parameters.
        return tfd.GaussianProcess(
          mean_fn=lambda x: np.mean(Y_train),
          kernel=kernel,
          index_points=X_train,
          observation_noise_variance=observation_noise_variance)
    
    return build_gp_internal


def prior_joint_model(X_train, Y_train, prior_dict=None):
    
    if prior_dict==None:
        prior_dict = {
            'smooth_amplitude': tfd.LogNormal(loc=0., scale=np.float64(1)),
            'smooth_length_scale': tfd.LogNormal(loc=0., scale=np.float64(1)),
            'periodic_amplitude': tfd.LogNormal(loc=0., scale=np.float64(1)),
            'periodic_length_scale': tfd.LogNormal(loc=0., scale=np.float64(1)),
            'periodic_period': tfd.LogNormal(loc=0., scale=np.float64(1)),
            'periodic_local_amplitude': tfd.LogNormal(loc=0., scale=np.float64(1)),
            'periodic_local_length_scale': tfd.LogNormal(loc=0., scale=np.float64(1)),
            'global_periodic_amplitude': tfd.LogNormal(loc=0., scale=np.float64(1)),
            'global_periodic_length_scale': tfd.LogNormal(loc=0., scale=np.float64(1)),
            'global_periodic_period': tfd.LogNormal(loc=2., scale=np.float64(1)),
            'irregular_amplitude': tfd.LogNormal(loc=0., scale=np.float64(1)),
            'irregular_length_scale': tfd.LogNormal(loc=0., scale=np.float64(1)),
            'irregular_scale_mixture': tfd.LogNormal(loc=0., scale=np.float64(1)),
            'matern_onehalf_amplitude': tfd.LogNormal(loc=0., scale=np.float64(1)),
            'matern_onehalf_length_scale': tfd.LogNormal(loc=0., scale=np.float64(1)),
            'matern_threehalves_amplitude': tfd.LogNormal(loc=0., scale=np.float64(1)),
            'matern_threehalves_length_scale': tfd.LogNormal(loc=0., scale=np.float64(1)),
            'matern_fivehalves_amplitude': tfd.LogNormal(loc=0., scale=np.float64(1)),
            'matern_fivehalves_length_scale': tfd.LogNormal(loc=0., scale=np.float64(1)),
            'observation_noise_variance': tfd.LogNormal(loc=0., scale=np.float64(1)),
        }
    
    prior_gp = build_gp(X_train, Y_train)
    prior_dict['observations'] = prior_gp
    
    return tfd.JointDistributionNamed(prior_dict)


def target_log_prob_from_joint(joint, Y_train):

    # Use `tf.function` to trace the loss for more efficient evaluation.
    @tf.function(autograph=False, experimental_compile=False)
    def target_log_prob(smooth_amplitude,
                       smooth_length_scale,
                       periodic_amplitude,
                       periodic_length_scale,
                       periodic_period,
                       periodic_local_amplitude,
                       periodic_local_length_scale,
                       global_periodic_amplitude,
                       global_periodic_length_scale,
                       global_periodic_period,
                       irregular_amplitude,
                       irregular_length_scale,
                       irregular_scale_mixture,
                       matern_onehalf_amplitude,
                       matern_onehalf_length_scale,
                       matern_threehalves_amplitude,
                       matern_threehalves_length_scale,
                       matern_fivehalves_amplitude,
                       matern_fivehalves_length_scale,
                       observation_noise_variance):
        
        return joint.log_prob({
          'smooth_amplitude': smooth_amplitude,
          'smooth_length_scale': smooth_length_scale,
          'periodic_amplitude': periodic_amplitude,
          'periodic_length_scale': periodic_length_scale,
          'periodic_period': periodic_period,
          'periodic_local_amplitude': periodic_local_amplitude,
          'periodic_local_length_scale': periodic_local_length_scale,
          'global_periodic_amplitude': global_periodic_amplitude,
          'global_periodic_length_scale': global_periodic_length_scale,
          'global_periodic_period': global_periodic_period,
          'irregular_amplitude': irregular_amplitude,
          'irregular_length_scale': irregular_length_scale,
          'irregular_scale_mixture': irregular_scale_mixture,
          'matern_onehalf_amplitude': matern_onehalf_amplitude,
          'matern_onehalf_length_scale': matern_onehalf_length_scale,
          'matern_threehalves_amplitude': matern_threehalves_amplitude,
          'matern_threehalves_length_scale': matern_threehalves_length_scale,
          'matern_fivehalves_amplitude': matern_fivehalves_amplitude,
          'matern_fivehalves_length_scale': matern_fivehalves_length_scale,
          'observation_noise_variance': observation_noise_variance,
          'observations': Y_train
        })
    
    return target_log_prob


with tf.device('/gpu:0'):
    # Speed up sampling by tracing with `tf.function`.
    @tf.function(autograph=False, experimental_compile=False)#, experimental_relax_shapes=True)
    def do_sampling(adaptive_sampler, initial_state, num_results=tf.constant(100), num_burnin_steps=tf.constant(500), parallel_iterations=tf.constant(10)):

        return tfp.mcmc.sample_chain(
          kernel=adaptive_sampler,
          current_state=initial_state,
          num_results=num_results,
          num_burnin_steps=num_burnin_steps,
          parallel_iterations=parallel_iterations,
          trace_fn=lambda current_state, kernel_results: kernel_results)
    

def theta_to_posterior(theta):
    
    if theta is None:
        return None
    
    posterior = {
        'smooth_amplitude': tfd.Empirical(theta[0]),
        'smooth_length_scale': tfd.Empirical(theta[1]),
        'periodic_amplitude': tfd.Empirical(theta[2]),
        'periodic_length_scale': tfd.Empirical(theta[3]),
        'periodic_period': tfd.Empirical(theta[4]),
        'periodic_local_amplitude': tfd.Empirical(theta[5]),
        'periodic_local_length_scale': tfd.Empirical(theta[6]),
        'global_periodic_amplitude': tfd.Empirical(theta[7]),
        'global_periodic_length_scale': tfd.Empirical(theta[8]),
        'global_periodic_period': tfd.Empirical(theta[9]),
        'irregular_amplitude': tfd.Empirical(theta[10]),
        'irregular_length_scale': tfd.Empirical(theta[11]),
        'irregular_scale_mixture': tfd.Empirical(theta[12]),
        'matern_onehalf_amplitude': tfd.Empirical(theta[13]),
        'matern_onehalf_length_scale': tfd.Empirical(theta[14]),
        'matern_threehalves_amplitude': tfd.Empirical(theta[15]),
        'matern_threehalves_length_scale': tfd.Empirical(theta[16]),
        'matern_fivehalves_amplitude': tfd.Empirical(theta[17]),
        'matern_fivehalves_length_scale': tfd.Empirical(theta[18]),
        'observation_noise_variance': tfd.Empirical(theta[19]),
    }
    
    return posterior


def sample(X, Y, step_size, num_results, num_leapfrog_steps, num_burnin_steps, constrain_positive, target_accept_prob, parallel_iterations, theta=None):
    
    posterior = theta_to_posterior(theta)
    
    # Create Joint Model
    joint = prior_joint_model(X, Y, posterior)
    
    # Target Log Prob
    target_log_prob = target_log_prob_from_joint(joint, Y)

    # Create sampler
    sampler = tfp.mcmc.TransformedTransitionKernel(
        tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob,
            step_size=tf.cast(step_size, tf.float64),
            num_leapfrog_steps=num_leapfrog_steps),
            bijector=[constrain_positive]*20)

    adaptive_sampler = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=sampler,
        num_adaptation_steps=int(0.8 * num_burnin_steps),
        target_accept_prob=tf.cast(target_accept_prob, tf.float64))

    # Initial States for pyhisical walk
    if theta == None:
        nitial_state = tf.constant([1.]*20, tf.float64)
        initial_state = [tf.cast(x, tf.float64) for x in [1.]*20]
    else:
        initial_state = [tf.cast(x[0], tf.float64) for x in theta]
    
    samples, kernel_results = do_sampling(adaptive_sampler,
                                              initial_state,
                                              num_results=num_results,
                                              num_burnin_steps=num_burnin_steps,
                                              parallel_iterations=parallel_iterations)
    
    return samples, kernel_results


def simulate_slice(sslice, theta=None, padding=27_000//48//2, n=100, num_leapfrog_steps=8, step_size=0.1, num_results=100,
                   num_burnin_steps=200, parallel_iterations=10, target_accept_prob=0.75, predictive_noise_variance=0., num_predictive_points=200):
    
    
    # Preparation
    X=sslice["Recording"]["ContractionNoNorm"]
    Y=sslice["Recording"]["RrInterval"]
    X = np.array(X).astype(float).reshape(-1, 1)
    Y = np.array(Y)
    
    X = tf.constant(X)
    Y = tf.constant(Y)
    
    constrain_positive = tfb.Shift(np.finfo(np.float64).tiny)(tfb.Exp())
    
    for i in range(5):
        try:
            samples, kernel_results = sample(X, Y,
                                     step_size,
                                     num_results,
                                     num_leapfrog_steps,
                                     num_burnin_steps,
                                     constrain_positive,
                                     target_accept_prob,
                                     parallel_iterations,
                                     theta=theta)
            break
        except tf.errors.InvalidArgumentError:
            if i >= 4:
                print(f"Cholesky decompositon error detected 6 times now. Trying one last time with inital priors.")
                samples, kernel_results = sample(X, Y,
                                                 step_size,
                                                 num_results,
                                                 num_leapfrog_steps,
                                                 num_burnin_steps,
                                                 constrain_positive,
                                                 target_accept_prob,
                                                 parallel_iterations,
                                                 theta=None)
                break
            print(f"Cholesky decompositon error detected in run {i+1}. Retrying.")                            

    (smooth_amplitude_samples,
    smooth_length_scale_samples,
    periodic_amplitude_samples,
    periodic_length_scale_samples,
    periodic_period_samples,
    periodic_local_amplitude_samples,
    periodic_local_length_scale_samples,
    global_periodic_amplitude_samples,
    global_periodic_length_scale_samples,
    global_periodic_period_samples,
    irregular_amplitude_samples,
    irregular_length_scale_samples,
    irregular_scale_mixture_samples,
    matern_onehalf_amplitude_samples,
    matern_onehalf_scale_samples,
    matern_threehalves_amplitude_samples,
    matern_threehalves_scale_samples,
    matern_fivehalves_amplitude_samples,
    matern_fivehalves_scale_samples,
    observation_noise_variance_samples) = samples
    
    # Given the posterior parameter and thus kernel samples, we can use that to fit the GP and predict for the posterior points
    predictive_index_points_ = np.linspace(0, padding-2, num_predictive_points, dtype=np.float64).reshape(-1, 1)
    
    with tf.device('/cpu:0'):

        # The sampled hyperparams have a leading batch dimension, `[num_results, ...]`,
        # so they construct a *batch* of kernels.
        batch_of_posterior_kernels = get_kernel_bayesian(smooth_amplitude_samples,
                                                    smooth_length_scale_samples,
                                                    periodic_amplitude_samples,
                                                    periodic_length_scale_samples,
                                                    periodic_period_samples,
                                                    periodic_local_amplitude_samples,
                                                    periodic_local_length_scale_samples,
                                                    global_periodic_amplitude_samples,
                                                    global_periodic_length_scale_samples,
                                                    global_periodic_period_samples,
                                                    irregular_amplitude_samples,
                                                    irregular_length_scale_samples,
                                                    irregular_scale_mixture_samples,
                                                    matern_onehalf_amplitude_samples,
                                                    matern_onehalf_scale_samples,
                                                    matern_threehalves_amplitude_samples,
                                                    matern_threehalves_scale_samples,
                                                    matern_fivehalves_amplitude_samples,
                                                    matern_fivehalves_scale_samples)

        # The batch of kernels creates a batch of GP predictive models, one for each
        # posterior sample.
        batch_gprm = tfd.GaussianProcessRegressionModel(
            mean_fn=lambda x: np.mean(Y),
            kernel=batch_of_posterior_kernels,
            index_points=predictive_index_points_,
            observation_index_points=X,
            observations=Y,
            observation_noise_variance=observation_noise_variance_samples,
            predictive_noise_variance=predictive_noise_variance)

        # To construct the marginal predictive distribution, we average with uniform
        # weight over the posterior samples.
        predictive_gprm = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(logits=tf.zeros([num_results])),
            components_distribution=batch_gprm)

        simulated = predictive_gprm.sample(n)
    
    return simulated, samples


def remove_outliers_in_recordings(recordings, in_seconds=True, low_rri: int = 300, high_rri: int = 2000, ectopic_beats_removal_method: str = "kamath"):
    
    cleared_recordings = []
    recordings_deepcopy = copy.deepcopy(recordings)
    denom = 1000 if in_seconds else 1
    
    for i, recording in enumerate(recordings_deepcopy):
        
        rrinterval = hrv.remove_outliers(recording["Recording"]["RrInterval"],
                                                               low_rri=low_rri, high_rri=high_rri, verbose=False)
        
        rrinterval = hrv.remove_ectopic_beats(rrinterval,
                                        method=ectopic_beats_removal_method, verbose=False)
        
        recording["Recording"]["RrInterval"] = np.array(rrinterval) / denom
        recording["Recording"].dropna(inplace=True)
        cleared_recordings.append(recording)
        
    return cleared_recordings


def c_splice_lod_constant_by_number(lod, n=48):
    spliced_recordings = []
    
    for i, recording in enumerate(lod):
        
        splices = np.array_split(recording["Recording"], n)
        
        for splice in splices:
            recording_deepcopy = copy.deepcopy(recording)
            recording_deepcopy["Recording"] = splice
            recording_deepcopy["Recording"]["ContractionNoNorm"] = list(range(len(recording_deepcopy["Recording"]["ContractionNoNorm"])))
            spliced_recordings.append(recording_deepcopy)
    
    return spliced_recordings


def simulation_generator(recordings, padding=27_000//48//2,
                                     n=100,
                                     num_leapfrog_steps=8,
                                     step_size=0.1,
                                     num_results=100,
                                     num_burnin_steps=200,
                                     parallel_iterations=10,
                                     target_accept_prob=0.75,
                                     predictive_noise_variance=0.,
                                     num_predictive_points=200):
    
    theta = None
    
    for recording in recordings:
        simulated_data, theta = simulate_slice(recording,
                                               theta=theta,
                                               padding=padding,
                                               n=n,
                                               num_leapfrog_steps=num_leapfrog_steps,
                                               step_size=step_size,
                                               num_results=num_results,
                                               num_burnin_steps=num_burnin_steps,
                                               parallel_iterations=parallel_iterations,
                                               target_accept_prob=target_accept_prob,
                                               predictive_noise_variance=predictive_noise_variance,
                                               num_predictive_points=num_predictive_points)
        yield recording, simulated_data
        

def file_len(fname):
    p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE, 
                                              stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])


def append_df_list_to_files(dfs, basedir, set_selector="train", padding=200, append=False, start_index=0):

    append_df_list_to_file_feature(dfs, basedir, set_selector, append, start_index)
    append_df_list_to_file_deep(dfs, basedir, set_selector, padding, append, start_index)
    

def append_df_list_to_file_feature(dfs, basedir, set_selector="train", append=False, start_index=0):
    
    header = not append
    mode = 'a' if append else 'w'
    indeces = list(range(start_index, start_index+len(dfs)))
    
    for df in dfs:
        df["Recording"]["RrInterval"] = df["Recording"]["RrInterval"] * 1000.0
        
    dfs_classification = utils.recordings_to_feature_dataframe(dfs, classification=True)
    dfs_regression = utils.recordings_to_feature_dataframe(dfs, classification=False)
    
    dfs_classification.index = indeces
    dfs_regression.index = indeces
    
    dfs_classification.to_csv(f'{basedir}simulated_gdansk_constant_features_classification_milliseconds_{set_selector}.csv', mode=mode, header=header)
    dfs_regression.to_csv(f'{basedir}simulated_gdansk_constant_features_regression_milliseconds_{set_selector}.csv', mode=mode, header=header)
    

def append_df_list_to_file_deep(dfs, basedir, set_selector="train", padding=200, append=False, start_index=0):
    
    header = not append
    mode = 'a' if append else 'w'
    indeces = list(range(start_index, start_index+len(dfs)))
    
    dfs_classification = utils.recordings_to_deep_dataframe(dfs, pad_length=padding, classification=True, gdansk=True)
    dfs_regression = utils.recordings_to_deep_dataframe(dfs, pad_length=padding, classification=False, gdansk=True)
    
    dfs_classification.index = indeces
    dfs_regression.index = indeces
    
    dfs_classification.to_csv(f'{basedir}simulated_gdansk_constant_deep_classification_seconds_{set_selector}.csv', mode=mode, header=header)
    dfs_regression.to_csv(f'{basedir}simulated_gdansk_constant_deep_regression_seconds_{set_selector}.csv', mode=mode, header=header)
    

def tensor_1d_to_recording(tensor, start_index=0):
    
    series = pd.Series(tensor)
    indeces = list(range(start_index, start_index+len(series)))
    
    df = pd.DataFrame({"ContractionNo": list(range(len(series))),
                       "ContractionNoNorm": list(range(len(series))),
                       "RrInterval": series})
    
    df.index = indeces

    return df


def simulate_data(orig_recordings,
                  set_selector="train",
                  basedir="../data/preprocessed/",
                  padding=27_000//48//2,
                  num_leapfrog_steps=8,
                  step_size=0.1,
                  n=10,
                  num_results=100,
                  num_burnin_steps=100,
                  parallel_iterations=10,
                  target_accept_prob=0.75,
                  predictive_noise_variance=0.,
                  num_predictive_points=200,
                  stop_early=None):
    
    path_feature_classification = f"{basedir}simulated_gdansk_constant_features_classification_milliseconds_{set_selector}.csv"
    path_feature_regression = f"{basedir}simulated_gdansk_constant_features_regression_milliseconds_{set_selector}.csv"
    path_deep_classification = f"{basedir}simulated_gdansk_constant_deep_classification_seconds_{set_selector}.csv"
    path_deep_regression = f"{basedir}simulated_gdansk_constant_deep_regression_seconds_{set_selector}.csv"
    
    output_paths = [path_feature_classification, path_feature_regression, path_deep_classification, path_deep_regression]
    
    # Determine where to start
    start = 0
    if all(map(path.exists, output_paths)):
        file_lengths = list(map(file_len, output_paths))
        if len(set(file_lengths)) == 1:
            start = list(file_lengths)[0]//n
    
    append = start != 0
    total_count = len(orig_recordings)
    
    print(f"Processing {start+1}/{total_count}")
    
    # If not everything is sane, raise an exception
    if os.path.exists(path_feature_classification) and start == 0:
        raise Exception(f"For safety reasons, file {path_feature_classification} must be deleted manually.")
        
    if os.path.exists(path_feature_regression) and start == 0:
        raise Exception(f"For safety reasons, file {path_feature_regression} must be deleted manually.")
        
    if os.path.exists(path_deep_classification) and start == 0:
        raise Exception(f"For safety reasons, file {path_deep_classification} must be deleted manually.")
        
    if os.path.exists(path_deep_regression) and start == 0:
        raise Exception(f"For safety reasons, file {path_deep_regression} must be deleted manually.")
    
    orig_recordings = orig_recordings[start:]
    
    start_time = time.time()
    print(f"Current time is: {datetime.now()}")

    for i, res in enumerate(simulation_generator(orig_recordings, n=n)):

        orig_recording, simulated_data = res
        
        print("Data simulated, creating list of dataframes.")
        
        simulated_dfs = []
        
        for j in range(simulated_data.shape[0]):
            
            simulated = simulated_data[j,:]
            
            recording_deepcopy = copy.deepcopy(orig_recording)
            recording_deepcopy["Recording"] = tensor_1d_to_recording(simulated)
            simulated_dfs.append(recording_deepcopy)
        
        print("List of dataframes created. Saving results to file.")
        
        index_continue = start*n + i*n
        append_df_list_to_files(simulated_dfs, basedir, set_selector, padding=200, append=append, start_index=index_continue)
        
        del simulated_dfs, orig_recording, simulated_data

        time_diff = time.time() - start_time
        time_remaining_in_hours = (time_diff*(len(orig_recordings)-i))//3600
        
        print(f"Time taken: {round(time_diff)} seconds")
        print(f"Estimated time remaining: {round(time_remaining_in_hours)} hours")
        print(f"Estimated time finished: {datetime.now() + timedelta(hours=time_remaining_in_hours)}")
        print(f"Early stopping progress: {i+1}/{stop_early}")
        print("\n")
        print(f"Processing {start+(i+2)}/{total_count} next, if available, or end.")
        
        if stop_early is not None and i+2 > stop_early:
            break
        
        append=True
        
        start_time = time.time()
        print(f"Current time is: {datetime.now()}")
