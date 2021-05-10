import pandas as pd
from zipfile import ZipFile

def read_lastfm(zip_name = "lastfm-dataset-1K.zip"):
    with ZipFile(zip_name, 'r') as z:
        folder = "lastfm-dataset-1K/"
        
        #Read song csv
        song_file = folder +"userid-timestamp-artid-artname-traid-traname.tsv"
        songs = pd.read_csv(z.open(song_file), sep='\t', error_bad_lines=False, 
                        header=None, parse_dates=[1], 
                        names=['user_id', 'timestamp', 'artist_id', 'artist_name', 
                        'track_id', 'track_name'],
                        dtype={0:'category', 2:'category',3:'category',4:'category',5:'category'})
        # songs.columns = ['user_id', 'timestamp', 'artist_id', 'artist_name', 'track_id', 'track_name']
        
        #Read user csv
        user_file = folder +"userid-profile.tsv"
        column_types = {'#id': 'category', 'gender':'category', 'age':'float32', 'country':'category'}
        users = pd.read_csv(z.open(user_file), sep='\t', parse_dates=[4], dtype=column_types) \
                    .rename(columns={'#id': 'user_id'})
    return songs, users
