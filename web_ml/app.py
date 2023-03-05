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
annoy_index.save('nearest_neightbor_graph.ann')
annoy_index = AnnoyIndex(audio_dim, 'angular')
annoy_index.load('nearest_neightbor_graph.ann')

# Use this list for reference of the music types in the dataset
musics = {}
for index in range(len(music_dataset[:1000])):
    sample = music_dataset[index]
    music_labels = [music_dict[idx] for idx in sample['label']]
    musics[index] = music_labels

with open("data/musics/ytb_musics_dict", "r") as f:
    ytb_musics_dict = json.load(f)

@app.route('/', methods=['GET','POST'])
def recommend():
    """ Use the nearest neighbors of the music liked by the user by using ANNOY. Rule-based chatbot (either recommends music by genre or from a particular artist.)
    """
    ytb_df = pd.Series(ytb_musics_dict)
    if request.method == 'POST':
        #print(request.form)
        music_input = request.form['music_request']
        #print(music_input)
        nns_index = annoy_index.get_nns_by_item(19, 10)
        recommends = []
        for index in nns_index:
            sample = music_dataset[index]
            music_labels = [music_dict[idx] for idx in sample['label']]
            print(ytb_df[index])
            url = "".join(["https://www.youtube.com/watch?v=",sample["video_id"].decode('utf-8')])
            recommends.append([index, music_labels, url, sample['start_time'], sample['end_time'], ytb_df[index]])
        playlist = pd.DataFrame(recommends, columns=['index', 'label', 'video_id', 'start_time', 'end_time', 'title'])
        return render_template("recommend.html", recommends =recommends, musics = musics, playlist=playlist, music_input=music_input)
    
    if request.method == 'GET':
        music_input = request.args.get('musics')
        print(music_input)
        nns_index = annoy_index.get_nns_by_item(19, 10)
        recommends = []

        for index in nns_index:
            sample = music_dataset[index]
            music_labels = [music_dict[idx] for idx in sample['label']]
            url = "".join(["https://www.youtube.com/watch?v=",sample["video_id"].decode('utf-8')])
            
            recommends.append([index, music_labels, url, sample['start_time'], sample['end_time'], ytb_df[index]])
        playlist = pd.DataFrame(recommends, columns=['index', 'label', 'video_id', 'start_time', 'end_time', 'title'])
        return render_template("request.html", recommends=recommends, musics = musics, playlist=playlist)

if __name__ == '__main__':
    app.run(debug=True)
