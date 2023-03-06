import random
import pandas as pd
from flask import jsonify, request, Flask, render_template
import json
from ast import literal_eval
from annoy import AnnoyIndex
import nltk
import pandas as pd
from wit import Wit

nltk.download('punkt')

app = Flask(__name__)

witTOKEN = "ALYL67J3HEMFL3SFQZDHLKO6JZYH7XBK"
witmodel = Wit( witTOKEN )

messages = []

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

def extract_entity(nlp_data):
    if nlp_data['entities']: 
        # get the best entity (movie with highest confidence AND confidence>0.5)
        entity_list = []
        currconf = 0.5
        for entity_body in nlp_data['entities'].values():
            print(entity_body)
            if entity_body[0]['confidence'] > currconf:
                entity_list.append(entity_body[0]['name'])

        return entity_list


@app.route('/chat', methods=["POST"])
def chat():
    user_input = request.json["user_input"]
    print(user_input)
    entities = extract_entity(witmodel.message(user_input))
    intents = witmodel.message(user_input)['intents']
    bot_answer = '';
    if entities:
        for entitiy in entities:
            if entitiy == 'hi':
                hi_res = ["Nice to meet you !", "How you're doing ?", "How's everything?", "Look what the cat dragged in!", "Hey, boo"]
                bot_answer += random.choice(hi_res)
            elif entitiy == 'health':
                health_res = ["I am doing great, thanks for asking.", "Hanging in there, how 'bout you?", "Pretty peachy.", "Not bad for a bot lol !"]
                bot_answer += random.choice(health_res)
            elif entitiy == 'health_response':
                health_res_res = ["That's great to hear !", "You deserve it !"]
                bot_answer += random.choice(health_res_res)
            elif entitiy == 'bye':
                bye_res = ["Farewell", "Catch you later", "Take care", "Goodbye !", "Bye boo !"]
                bot_answer += random.choice(bye_res)
            else :
                bot_answer += "So you want to listen to " + entitiy + " ? Please choose a song below and I will gladly recommand you some good songs"
    else:
        bot_answer += "Sorry. I didn't understand. Can you rephrase it please ?"
        
    # add user input as message from user
    messages.append({"sender": "user", "text": user_input})
    
    messages.append({"sender": "chatbot", "text": bot_answer})
    
    print(messages)
    return jsonify([{"sender": "chatbot", "text": bot_answer},{"sender": "user", "text": user_input}])
    
if __name__ == '__main__':
    text="i want to  listen to  rap"
    print(text)
    entities = extract_entity(witmodel.message(text))
    intents = witmodel.message(text)['intents']

    if intents:
        if entities:
            entity = entities
            print(entity)
    app.run(debug=True)