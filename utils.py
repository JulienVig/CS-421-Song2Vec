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
        
        #Read user csv
        user_file = folder +"userid-profile.tsv"
        column_types = {'#id': 'category', 'gender':'category', 'age':'float32', 'country':'category'}
        users = pd.read_csv(z.open(user_file), sep='\t', parse_dates=[4], dtype=column_types) \
                    .rename(columns={'#id': 'user_id'})
    return songs, users

def build_vocab(model):
    emb_vectors = {}
    for n in model.wv.index_to_key:
        emb_vectors[n] = model.wv[n]
    return emb_vectors

def get_embeddings(df, , **kwargs):
    df = df.sort_values("timestamp")
    df = df[~df.track_name.isna()]
    df["song_id"]= df.artist_name.cat.codes.astype("int64") * df.track_name.nunique() \
                        + df.track_name.cat.codes
    print(f"Number of entries: {len(df)}")
    assert len(df.query("song_id < 0")) == 0 #Check underflow
    
    document = df.groupby("user_id").agg(sentences=("song_id", list))
    model = Word2Vec(document.sentences.values, , **kwargs)
    
    emb_vectors = build_vocab(model)
    print(f"Number of song embeddings: {len(emb_vectors)}")
    return df, emb_vectors, model

def load_model(filename):
    model = Word2Vec.load(filename)
    emb_vectors = build_vocab(model)
    return emb_vectors, model