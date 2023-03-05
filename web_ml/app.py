import pandas as pd
import numpy as np
import random
from flask import request, Flask, render_template
import json
from ast import literal_eval
from annoy import AnnoyIndex
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import bs4 as bs
import urllib.request
import pandas as pd
nltk.download('punkt')

app = Flask(__name__)

"""def chatbot_answer(user_query):
    Measure the cosine similarity and take the second closest index because the first index is the user query
    user_query
    
    cat_data = urllib.request.urlopen('https://simple.wikipedia.org/wiki/Cat').read()
    cat_data_paragraphs  = bs.BeautifulSoup(cat_data,'lxml').find_all('p')
    txt = ''
    for p in cat_data_paragraphs:
        txt += p.text.lower()
    sentences = nltk.sent_tokenize(txt)
    sentences.append(user_query)
    vectorizer = TfidfVectorizer()
    sentences_vectors = vectorizer.fit_transform(sentences)
    
    vector_values = cosine_similarity(sentences_vectors[-1], sentences_vectors)
    answer = sentences[vector_values.argsort()[0][-2]]
    #Final check to make sure there are result present. If all the result are 0, means the text input by us are not captured in the corpus
    input_check = vector_values.flatten()
    input_check.sort()
    
    if input_check[-2] == 0:
        return "Please Try again"
    else: 
        return answer
"""

@app.route('/', methods=['GET','POST'])
def recommend():
    """ Use the nearest neighbors of the music liked by the user by using ANNOY
    """
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
    
    if request.method == 'POST':
        # cell for search from specific music 
        music_input = request.form['music_request']
        print(music_input)
     
        nns_index = annoy_index.get_nns_by_item(19, 10)
        recommends = []
        for index in nns_index:
            sample = music_dataset[index]
            music_labels = [music_dict[idx] for idx in sample['label']]
            recommends.append([index, music_labels, sample['video_id'], sample['start_time'], sample['end_time']])
        playlist = pd.DataFrame(recommends, columns=['index', 'label', 'video_id', 'start_time', 'end_time'])

        return render_template("recommend.html", recommends =recommends, musics = musics, playlist=playlist, music_input=music_input)
    if request.method == 'GET':
        # cell for search from specific music 
        music_input = request.args.get('musics')
        print(music_input)
        nns_index = annoy_index.get_nns_by_item(19, 10)
        recommends = []

        for index in nns_index:
            sample = music_dataset[index]
            music_labels = [music_dict[idx] for idx in sample['label']]
            recommends.append([index, music_labels, sample['video_id'], sample['start_time'], sample['end_time']])
        playlist = pd.DataFrame(recommends, columns=['index', 'label', 'video_id', 'start_time', 'end_time'])
        return render_template("request.html", recommends=recommends, musics = musics, playlist=playlist)

if __name__ == '__main__':
    app.run(debug=True)
