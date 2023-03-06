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


@app.route('/', methods=["GET", "POST"])
def recommend(music_input=None):
    """ Use the nearest neighbors of the music liked by the user by using ANNOY. Rule-based chatbot (either recommends music by genre or from a particular artist.)
    Choose a music (among the 2070) based on the genres you like:
    """
    if request.method == "GET" or request.method == "POST":
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
        print(music_req_l)
        recommends = []
        for index in nns_index:
            if type(ytb_df[index]) != list:
                recommends.append(ytb_df[index])
        return render_template("request.html", recommends =recommends, musics = musics)
    
if __name__ == '__main__':
    app.run(debug=True)