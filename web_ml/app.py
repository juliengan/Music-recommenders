import pandas as pd
from flask import request, Flask, render_template
import json
from ast import literal_eval
from annoy import AnnoyIndex
import nltk
import pandas as pd

nltk.download('punkt')

app = Flask(__name__)

class_labels = pd.read_csv('data/musics/class_labels_indices.csv')
music_dict = dict(zip(class_labels.index, class_labels.display_name))
with open('data/musics/music_set.json', 'r') as file:
    file_read = json.loads(file.read())
    music_dataset = literal_eval(file_read)
audio_dim = 1280
annoy_index = AnnoyIndex(audio_dim, 'angular')

for index in range(len(music_dataset[:1000])):
    vector = music_dataset[index]['data']
    annoy_index.add_item(index, vector)

annoy_index.build(50)
annoy_index.save('../models/nearest_neightbor_graph.ann')
annoy_index = AnnoyIndex(audio_dim, 'angular')
annoy_index.load('../models/nearest_neightbor_graph.ann')


@app.route('/')
def recommend(music_input=None):
    """ Use the nearest neighbors of the music liked by the user by using ANNOY. Rule-based chatbot (either recommends music by genre or from a particular artist.)
    """
    musics = {}
    for index in range(len(music_dataset[:1000])):
        sample = music_dataset[index]
        music_labels = [music_dict[idx] for idx in sample['label']]
        musics[index] = music_labels

    with open("data/musics/ytb_musics_dict", "r") as f:
        ytb_musics_dict = json.load(f)
    ytb_df = pd.Series(ytb_musics_dict)
    music_req_l = request.form.getlist("music_request")
    music_input = int(music_req_l[0]) if music_req_l else 0 
    nns_index = annoy_index.get_nns_by_item(music_input, 10)
    recommends = []
    for index in nns_index:
        sample = music_dataset[index]
        music_labels = [music_dict[idx] for idx in sample['label']]
        url = "".join(["https://www.youtube.com/watch?v=",sample["video_id"].decode('utf-8')])
        recommends.append([index, music_labels, url, sample['start_time'], sample['end_time'], ytb_df[index]])
    playlist = pd.DataFrame(recommends, columns=['index', 'label', 'video_id', 'start_time', 'end_time', 'title'])
    return render_template("request.html", recommends =recommends, musics = musics, playlist=playlist, music_input=music_input)
    
if __name__ == '__main__':
    app.run(debug=True)
