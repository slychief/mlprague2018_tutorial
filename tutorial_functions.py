import numpy as np
from joblib import Memory
import progressbar
from joblib import Memory
import pandas as pd

import keras
from matplotlib import pyplot as plt
from IPython.display import clear_output

from keras.models           import Model
from keras.layers           import Input, Lambda, Dense, Bidirectional
from keras.layers.recurrent import LSTM
from keras.layers.merge     import concatenate
from keras.optimizers       import Nadam
from keras import backend as K



def init(cachedir):
    global memory
    memory = Memory(cachedir=cachedir, verbose=0)

def get_playlist_metadata(spotify_client, playlists):

    for playlist in playlists:

        # get user and playlist_id from uri
        (_,_,user,_,playlist_id) = playlist["uri"].split(":")

        # retrieve playlist metadat from Spotify
        playlist_metadata = spotify_client.user_playlist(user        = user,
                                                         playlist_id = playlist_id)

        # extract relevant information
        playlist["user"]        = user
        playlist["playlist_id"] = playlist_id
        playlist["num_tracks"]  = playlist_metadata["tracks"]["total"]

        # initialize fields for further processing
        playlist["track_ids"]   = []
        
    return playlists


def get_track_ids(sp, playlists):

    # max Spotify batch size
    batch_size = 100

    # retrieve tracks for each playlist
    for playlist in playlists:

        # batch processing
        for offset in np.arange(0, playlist["num_tracks"], batch_size):

            limit = np.min([batch_size, playlist["num_tracks"] - offset])

            playlist_entries = sp.user_playlist_tracks(user        = playlist["user"],
                                                       playlist_id = playlist["playlist_id"], 
                                                       limit       = limit, 
                                                       offset      = offset,
                                                       fields      = ["items"])

            playlist["track_ids"].extend([entry["track"]["id"] for entry in playlist_entries["items"]])
            
    return playlists


def aggregate_metadata(raw_track_data):

    metadata = []

    for playlist_name, spotify_data in raw_track_data:

        track_metadata, album_metadata, artist_metadata, _, _ = spotify_data

        # get year of album release
        release_date = album_metadata["release_date"]

        if album_metadata["release_date_precision"] != "year":
            release_date = release_date.split("-")[0]

        # assamble metadata
        metadata.append([track_metadata["id"],
                         artist_metadata["name"], 
                         track_metadata["name"], 
                         album_metadata["name"],
                         album_metadata["label"],
                         track_metadata["duration_ms"],
                         track_metadata["popularity"],
                         release_date,
                         artist_metadata["genres"], 
                         playlist_name])

    metadata = pd.DataFrame(metadata, columns=["track_id", "artist_name", "title", "album_name", "label", 
                                               "duration", "popularity",  "year",  "genres", "playlist"])
    
    return metadata


def aggregate_features(seq_data, track_data, metadata, with_year=False, with_popularity=False):

    calc_statistical_moments = lambda x: np.concatenate([x.mean(axis=0), x.std(axis=0)])
    
    # sequential data
    segments = seq_data["segments"]
    sl       = len(segments)
    
    # MFCCs - 24 dimensions
    mfcc              = np.array([s["timbre"] for s in segments])
    mfcc              = calc_statistical_moments(mfcc)
    
    # Chroma / pitch classes - 24 dimensions
    chroma            = np.array([s["pitches"] for s in segments])
    chroma            = calc_statistical_moments(chroma)
    
    # maximum loudness values per segment - 2 dimensions
    loudness_max      = np.array([s["loudness_max"] for s in segments]).reshape((sl,1))
    loudness_max      = calc_statistical_moments(loudness_max)
    
    # offset of max loudness value within segment - 2 dimensions
    loudness_start    = np.array([s["loudness_start"] for s in segments]).reshape((sl,1))
    loudness_start    = calc_statistical_moments(loudness_start)
    
    # length of max loudness values within segment - 2 dimensions
    loudness_max_time = np.array([s["loudness_max_time"] for s in segments]).reshape((sl,1))
    loudness_max_time = calc_statistical_moments(loudness_max_time)
    
    # length of segment - 2 dimensions
    duration          = np.array([s["duration"] for s in segments]).reshape((sl,1))
    duration          = calc_statistical_moments(duration)
    
    # confidence of segment boundary detection - 2 dimensions
    confidence        = np.array([s["confidence"] for s in segments]).reshape((sl,1))
    confidence        = calc_statistical_moments(confidence)
    
    # concatenate sequential features
    sequential_features = np.concatenate([mfcc, chroma, loudness_max, loudness_start, 
                                          loudness_max_time, duration, confidence], axis=0)
    
    # track-based data
    track_features = [track_data[0]["acousticness"],     # acoustic or not?
                      track_data[0]["danceability"],     # danceable?
                      track_data[0]["energy"],           # energetic or calm?
                      track_data[0]["instrumentalness"], # is somebody singing?
                      track_data[0]["liveness"],         # live or studio?
                      track_data[0]["speechiness"],      # rap or singing?
                      track_data[0]["tempo"],            # slow or fast?
                      track_data[0]["time_signature"],   # 3/4, 4/4, 6/8, etc.
                      track_data[0]["valence"]]          # happy or sad?
    
    if with_year:
        track_features.append(int(metadata["year"]))
        
    if with_popularity:
        track_features.append(int(metadata["popularity"]))
        
    
    return np.concatenate([sequential_features, track_features], axis=0)


def aggregate_featuredata(raw_track_data, metadata):

    feature_data = []

    for i, (_, spotify_data) in enumerate(raw_track_data):

        _, _, _, f_sequential, f_trackbased = spotify_data

        feature_vec = aggregate_features(f_sequential, 
                                         f_trackbased, 
                                         metadata.iloc[i], 
                                         with_year       = True, 
                                         with_popularity = True)    

        feature_data.append(feature_vec)

    feature_data = np.asarray(feature_data)
    
    return feature_data
	
	
# updatable plot
# a minimal example (sort of)

class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show();

		
def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))
	
def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))
	
