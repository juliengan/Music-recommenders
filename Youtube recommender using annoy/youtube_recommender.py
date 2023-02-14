# %%
!pip3 install --no-cache-dir annoy
!pip3 install -r '../requirements.txt'
# %%
import json
from ast import literal_eval
import pandas as pd

# %%
class_labels = pd.read_csv('class_labels_indices.csv')
music_dict = dict(zip(class_labels.index, class_labels.display_name))

# %%
with open(r'../web_ml/data/musics/music_set.json', 'r') as file:
    file_read = json.loads(file.read())
    music_dataset = literal_eval(file_read)

# %%
len(music_dataset)

# %%
from annoy import AnnoyIndex

audio_dim = 1280
annoy_index = AnnoyIndex(audio_dim, 'angular')  
for index in range(len(music_dataset[:1000])):
    vector = music_dataset[index]['data']
    annoy_index.add_item(index, vector)

annoy_index.build(50)
annoy_index.save('nearest_neightbor_graph.ann')

# %%
annoy_index = AnnoyIndex(audio_dim, 'angular')
annoy_index.load('nearest_neightbor_graph.ann')

# %%
# cell for search from specific music 
nns_index = annoy_index.get_nns_by_item(7, 10)

# %%
recommends = []
for index in nns_index:
    sample = music_dataset[index]
    music_labels = [music_dict[idx] for idx in sample['label']]
    recommends.append([index, music_labels, sample['video_id'], sample['start_time'], sample['end_time']])

# %%
recommends

# %%
playlist = pd.DataFrame(recommends, columns=['index', 'label', 'video_id', 'start_time', 'end_time'])
print(playlist)

# %%
## Use this list for reference of the music types in the dataset
musics = {}
for index in range(len(music_dataset[:1000])):
    sample = music_dataset[index]
    music_labels = [music_dict[idx] for idx in sample['label']]
    musics[index] = music_labels

# %%
musics

# %%



