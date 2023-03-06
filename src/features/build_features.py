from pytube import YouTube
import json
from ast import literal_eval


def get_music_details(musics):
    """ Retrieves the titles of YouTube musics
    """
    with open('data/musics/music_set.json', 'r') as file:
        file_read = json.loads(file.read())
        music_dataset = literal_eval(file_read)
    ytb_musics_dict = dict(musics)
    for i in musics.items():
        url = "".join(["https://www.youtube.com/watch?v=",music_dataset[i[0]]["video_id"].decode('utf-8')])
        try:
            yt = YouTube(url)
            ytb_musics_dict[i[0]] = yt.title
            print("succeedeed")
        except Exception as e:
            print(f"An error occurred: {e}")

    with open("data/musics/ytb_musics_dict", "w") as f:
        json.dump(ytb_musics_dict, f)