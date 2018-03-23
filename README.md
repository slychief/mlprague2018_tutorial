# Deep Learning for Music Classification using Keras

(c) 2018 by

**Alexander Schindler**, AIT Austrian Institute of Technology<br>
http://ifs.tuwien.ac.at/~schindler

**Thomas Lidy**, Musimap<br>
https://www.linkedin.com/in/thomaslidy/

# Abstract

Deep Learning has re-entered the field of Machine Learning with a big bang. Tasks that were difficult and cumbersome to solve are now simple to implement and achieve amazingly high accuracies. Deep Learning has shown substantial impact recently also in the domain of audio recognition and music classification. This tutorial will give a general introduction to Neural Networks, Convolutional Neural Networks, plus important concepts such as (batch-)normalization, training epochs, activation, loss, optimization, pooling, dropout, model fine-tuning, etc. and a tutorial on how to apply all of this on tasks of audio and music recognition using the convenient Python Keras framework on top of Tensorflow. It will use T-SNE for visualization and also cover unsupervised learning.

# Tutorial Outline

**Part 1 - Deep Learning Basics**
  * Audio Processing Basics, 
  * History of Neural Networks
  * What is Deep Learning
  * Neural Network Concepts
  * Coding Examples

**Part 2 - Convolutional Neural Networks**
  * Difference CNN – RNN
  * How CNNs work (Layers, Filters, Pooling)
  * Application Domains and how to use in Music
  * Coding Examples

**Part 3 - Instrumental, Genre and Mood Analysis**
  * Large-scale Music AI at Musimap
  * Instrumental vs. Vocal Detection
  * Genre Recognition
  * Mood Detection
  * Coding Examples

**Part 4 - Advanced Deep Learning**
  * Similarity Retrieval
  * Siamese Networks
  * Learning Audio Representation from Tag Similarity
  * Coding Examples

# Tutorial Requirements

For the tutorials, we use iPython / Jupyter notebook, which allows to program and execute Python code interactively in the browser.

### Viewing Only

If you do not want to install anything, you can simply view the tutorials' content in your browser, by clicking on the tutorial's filenames listed below in the GIT file listing (above, resp. on https://github.com/... ).

The tutorial will open in your browser for viewing.

### Interactive Coding

If you want to follow the Tutorials by actually executing the code on your computer, please [install first the pre-requisites](#installation-of-pre-requisites) as described below.

After that, to run the tutorials go into the `mlprague2018_tutorial` folder and start from the command line:

`jupyter notebook`


# Installation of Pre-requisites

## Install Python 3.x

Note: On most Mac and Linux systems Python is already pre-installed. Check with `python --version` on the command line whether you have Python 2.7.x installed.

Otherwise install Python 3.5 from https://www.python.org/downloads/release/python-350/

## Install Python libraries:

### Mac, Linux or Windows

(on Windows leave out `sudo`)

Important note: If you have Python 2.x and 3.x installed in parallel, replace `pip` by `pip3` in the following commands:

```
sudo pip install jupyter
```

Try if you can open 
```
jupyter notebook
```
on the command line. 

Then download or clone the Tutorials from this GIT repository:

```
git clone https://github.com/slychief/mlprague2018_tutorial.git
```
or download https://github.com/slychief/mlprague2018_tutorial/archive/master.zip <br/>
unzip it and rename the folder to `mlprague2018_tutorial`.

Install the remaining Python libraries needed:

Either by:

```
sudo pip install Keras>=2.1.1 tensorflow scikit-learn>=0.18 pandas librosa spotipy matplotlib
```

or, if you downloaded or cloned this repository, by:

```
cd mlprague2018_tutorial
sudo pip install -r requirements.txt
```

### Optional for GPU computation

If you want to train your neural networks on your GPU (which is faster, but not necessarily needed for this tutorial),
you have to install the specific GPU version of Tensorflow:

```
sudo pip install tensorflow-gpu
```

and also install the following (not needed for the tutorials):

* [NVidia drivers](http://www.nvidia.com/Download/index.aspx?lang=en-us)
* [CUDA](https://developer.nvidia.com/cuda-downloads)
* [cuDNN](https://developer.nvidia.com/cudnn) (requires registration with Nvidia)


## Install Audio Decoder

In order to decode MP3 files (used in the MagnaTagAtune data set) you will need to install FFMpeg on your system.

* Linux: `sudo apt-get install ffmpeg`
* Mac: download FFMPeg for Mac: http://ffmpegmac.net and make sure ffmpeg is on PATH
* Windows: download https://github.com/tuwien-musicir/rp_extract/blob/master/bin/external/win/ffmpeg.exe and make sure it is on the PATH



### Prepared Datasets for Download

Please download the following data sets for this tutorial:

**GTZAN Music Speech Classification**

https://owncloud.tuwien.ac.at/index.php/s/JiBXUPZK9LImTHB (145MB)

**MagnaTagAtune**

https://owncloud.tuwien.ac.at/index.php/s/hivOGXKoUQtacbo (332MB)

These are prepared versions from the original datasets described below.


# Source Credits

## Python libraries

The following helper Python libraries are used in these tutorials:

* `audiofile_read.py` and `audio_spectrogram.py`: by Thomas Lidy and Alexander Schindler, taken from the [RP_extract](https://github.com/tuwien-musicir/rp_extract) git repository
* `wavio.py`: by Warren Weckesser

## Data Sources

The data sets we use in the tutorials are from the following sources:

* GTZAN Music Speech:
by George Tzanetakis
Collected for the purposes of music/speech discrimination. Consists of 128 tracks, each 30 seconds long. Each class (music/speech) has 64 examples in 22050Hz Mono 16-bit WAV audio format.
http://marsyasweb.appspot.com/download/data_sets/

* MagnaTagAtune:
http://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset
