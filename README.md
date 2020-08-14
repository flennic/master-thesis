# Master Thesis

Repository for the project in the course 732A64 Master's Thesis at Linköping University spring 2020.

## Abstract

Electrocardiography (ECG) is a non-invasive method used in medicine to track the electrical pulses sent by the heart. The time between two subsequent electrical impulses and hence the heartbeat of a subject, is referred to as an RR interval. Previous studies show that RR intervals can be used for identifying sleep patterns and cardiovascular diseases. Additional research indicates that RR intervals can be used to predict the cardiovascular age of a subject. This thesis investigates, if this assumption is true, based on two different datasets as well as simulated data based on Gaussian Processes. The datasets used are Holter recordings provided by the University of Gdańsk as well as a dataset provided by Physionet. The for mer represents a balanced dataset of recordings during nocturnal sleep of healthy subjects whereas the latter one describes an imbalanced dataset of records of a whole day of subjects that suffered from myocardial infarction. Feature-based models as well as a deep learning architecture called DeepSleep, based on a paper for sleep stage detection, are trained. The results show, that the prediction of a subject’s age, only based in RR intervals, is difficult. For the first dataset, the highest obtained test accuracy is 37.84 per cent, with a baseline of 18.23 per cent. For the second dataset, the highest obtained accuracy is 42.58 per cent with a baseline of 39.14 per cent. Furthermore, data is simulated by fitting Gaussian Processes to the first dataset and following a Bayesian approach by assuming a distribution for all hyperparameters of the kernel function in use. The distributions for the hyperparameters are continuously updated by fitting a Gaussian Process to a slices of around 2.5 minutes. Then, samples from the fitted Gaussian Process are taken as simulated data, handling impurity and padding. The  results show that the highest accuracy achieved is 31.12 per cent with a baseline of 18.23 per cent. Concludingly, the cardiovascular age prediction based on RR intervals is a difficult problem and complex handling of impurity does not necessarily improve the results.

## Purpose

This project is my Master thesis. The repository itself consists of several folders: ```report``` includes the written report with associated figures and results from the different models, the folder ```presentations``` includes the slides for the proposal seminar, the mid-term seminar and the final defense, `notes` includes the results in an Excel table as well as the proposal of the thesis, ```figures``` holds all the figures that I created for the thesis, either via draw.io or Jupyter notebooks for creating the figures, ```data``` holds the data provides by the University of Gdańsk and `notebooks` the code for running the models, where some parts are written in ordinary ```.py``` scripts and older versions and some other notebooks are available in a subfolder called ```playground```.

## Getting Started

1. Clone the repository: ```git clone git@github.com:flennic/master-thesis.git```.
2. Obtain the PhysioNet dataset. More information about that in the respective chapter.
3. For running the models, you will need one anaconda environment, if you also want to simulate the data using Gaussian Processes, you will need another environment as well. I recommend to install all packages for both environments and just separate them by using PyTroch and TensorFlow respectively:
    - Create your environment ```conda create --name myenv```.
    - Install required packages ```conda install jupyter-lab pandas numpy sklearn matplotlib seaborn xgboost```, you will also need ```pip install hrv-analysis```.
    - For running the models, you will need PyTorch: ```conda install pytorch torchvision cudatoolkit=10.2 -c pytorch```. You might adjust the CUDA version, depending on your hardware.
    - If you also want to simulate the data, you will need TensorFlow Probability. I recommend installing it into another environment: ```pip install --upgrade tensorflow-probability```.
    - **Note:** I've written down this part from my mind, so maybe so need to tweak some things manually. Especially installing the correct version of TensorFlow Probability wasn't that easy in the beginning of 2020.
4. You need to pre-pocess the data which you can do by using the notebook ```preprocessing.ipynb```. For preprocessing the data, depending on the model you want to run, there exist multiple versions and parameters which you can adjust in the notebook. If a combination is not available, the notebook should automatically stop at a specific point or it automatically adjusts wrongly set parameters. That holds for all notebooks. Make sure you adjust the paths. Depending on your system and the type of data, the preprocessing can take a while! The data will be saved automatically.
5. Now you should be able to run all feature-based and the deep learning models. The hyperparameters are set to work with 32GB of memory and 8GB of GPU memory which you might want to adjust. For the deep learning model working with PhysioNets complete time series, you will need 8GB of GPU memory if you do not want to make the model smaller, as the batch size is already set to 1 or 2.
6. For simulating the data, you can either use the notebook or the normal Python script. As it takes quite a while for the simulation (around a week), I'd recommend to not use the notebook. The Python code has a memory leak somewhere (or one of the underlying libraries), so it will eventually crash. Therefore, I programmed the script to be resumable, it will automatically simulate where it left of (but then starting with the initial priors again). An easy solution is to just run the script using ```while true; do python simulate_data_bayesian.py 60; sleep 1; done```.

## Obtaining and Processing the PhysioNet Dataset

Download the data set with: ```wget -r -N -c -np https://physionet.org/files/crisdb/1.0.0/```. Make sure you adjust the paths at the top of the notebooks. A thorough description of the data set can be found at https://physionet.org/content/crisdb/1.0.0/.

To make the preprocessing work, you must install the ```wfdb``` software. The quick start guide for Linux can be found here: https://archive.physionet.org/physiotools/wfdb-linux-quick-start.shtml

## License Information

See the attached license file for further notice.
