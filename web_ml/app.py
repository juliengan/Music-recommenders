import random
import pandas as pd
from flask import jsonify, request, Flask, render_template
import json
from ast import literal_eval
from annoy import AnnoyIndex
import nltk
import pandas as pd
from wit import Wit
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

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
stop_words = stopwords.words('english')


@app.route('/', methods=["GET", "POST"])
def recommend():
    """ Use the nearest neighbors of the music liked by the user by using ANNOY. Rule-based chatbot (either recommends music by genre or from a particular artist.)
    Choose a music (among the 2070) based on the genres you like:
    """
    if request.method == "GET" or request.method == "POST":
        return render_template("request.html")

def extract_entity(nlp_data):
    """get the best entity from Wit AI (music with highest confidence AND confidence>0.5)
    """
    if nlp_data['entities']: 
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
    bot_answer = ''
    if entities:
        for entitiy in entities:
            if entitiy == 'hi':
                hi_res = ["Nice to meet you !", "How you're doing ?", "How's everything?", "Look what the cat dragged in!", "Hey, boo"]
                bot_answer += random.choice(hi_res)
            elif entitiy == 'health':
                health_res = ["I am doing great, thanks for asking.", "Hanging in there, how 'bout you?", "Pretty peachy.", "Not bad for a bot lol !"]
                bot_answer += random.choice(health_res)
            elif entitiy == 'bye':
                bye_res = ["Farewell", "Catch you later", "Take care", "Goodbye !", "Bye boo !"]
                bot_answer += random.choice(bye_res)
            else :
                if user_input is None:
                    user_input = 'Hello'
                
                tokens = word_tokenize(user_input)
                filtered_tokens = [token for token in tokens if token.lower() not in stopwords.words('english')]
                stemmer = PorterStemmer()
                stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
                lemmatizer = WordNetLemmatizer()
                lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
                print("lemmatized user input:", lemmatized_tokens)
                
                musics = {}
                for index in range(len(music_dataset[:1000])):
                    sample = music_dataset[index]
                    music_labels = [music_dict[idx] for idx in sample['label']]
                    musics[index] = [string.lower() for string in music_labels] 

                # get all genres and filter lemmatized_tokens
                all_genres = musics.items()
                genres = []
                for genre_list in all_genres:
                    genres.extend(genre_list[1])
                genres = list(set(genres))
                lowered_genres = [string.lower() for string in genres]
                print("YoutuBot is based on the following genres",lowered_genres)
                filtered_input = []

                # Retrieve genre searched by the user
                for word in lemmatized_tokens:
                    for genre in lowered_genres:
                        if word in genre:
                            filtered_input.append(word)

                if len(filtered_input) > 0:
                    associated_musics = []
                    for k,v in musics.items():
                        associated_musics.append({k: v for w in v if filtered_input[0] in w})

                    # Retrieve musics of that genre
                    associated_musics = [d for d in associated_musics if d]
                    first_value = next(iter(associated_musics[0].keys()))
                else:
                    first_value = 0
                    bot_answer += "Sorry. I didn't understand. Can you rephrase it please ?"
                     
                print(first_value)
                with open("data/musics/ytb_musics_dict", "r") as f:
                    ytb_musics_dict = json.load(f)
                ytb_df = pd.Series(ytb_musics_dict)
                nns_index = annoy_index.get_nns_by_item(first_value, 10)
                recommends = []
                for index in nns_index:
                    if type(ytb_df[index]) != list:
                        recommends.append(ytb_df[index])
                bot_answer += "You should try to listen those songs : " + "".join(recommends)
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