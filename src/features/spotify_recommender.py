import Recommenders as Recommenders
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.model_selection import train_test_split
import pandas as pd

cid = '74a88342b13c4acfb29d53ba0b8a2540'
secret = '52c115ad7ecf4c76974e6341fd02f003'

#Authentication 
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)

playlist_link = "https://open.spotify.com/playlist/4XhkvMbp57gf2QpZPwPq3z"
playlist_URI = playlist_link.split("/")[-1].split("?")[0]
track_uris = [x["track"]["uri"] for x in sp.playlist_tracks(playlist_URI)["items"]]

playlist = sp.user_playlist_tracks('spotify', '4XhkvMbp57gf2QpZPwPq3z') 
songs = playlist['items']
df = pd.DataFrame(songs)
print(df.columns)
df.to_csv('Songs.csv', sep=';', encoding='utf-8', index=True)

artist_name =list()
track_name=list()
track_id =list()
popularity = list()
genres = list()
nb_followers = list()
album = list()
track_uri= []
user_id=[]
for i, item in enumerate(playlist['items']):
    track = item['track']
    #Main Artist
    artist_uri = track["artists"][0]["uri"]
    #song_df["artist_uri"][i] = artist_uri
    artist_info = sp.artist(artist_uri)
    artist_name.append(track['artists'][0]['name'])
    track_name.append(track['name'])
    track_id.append(track['id'])
    popularity.append(track['popularity'])
    genres.append(artist_info['genres'])
    nb_followers.append(artist_info["followers"]["total"])
    album.append(track["album"]["name"])
    track_uri.append(track["uri"])
    user_id.append(1194339014)


song_df = pd.DataFrame({'user_id' : user_id,'artist_name':artist_name,'track_name':track_name,'track_id':track_id, 
'popularity':popularity, 'genres':genres, "nb_followers":nb_followers, "album": album, 
"track_uri":track_uri, "audio_features": sp.audio_features(track_uri)})

train_data, test_data = train_test_split(song_df, test_size = 0.20, random_state=0)
print(train_data)
pm = Recommenders.popularity_recommender_py()
pm.create(train_data, 'user_id', 'track_name')
users = song_df['user_id'].unique()
user_id = users[0]
recom = pm.recommend(user_id)
print(recom)
train_data, test_data = train_test_split(song_df, test_size = 0.20, random_state=0)
print(train_data)
pm = Recommenders.item_similarity_recommender_py()
pm.create(train_data, 'user_id', 'track_name')
users = song_df['user_id'].unique()
user_id = users[0]
print(pm.recommend(user_id))