<h1>YoutuBot</h1>
==============================

Chatbot that recommends musics from YouTube 
==============================


<h2>Overall project steps</h2>
<br>
<h3>1. Retrieve YouTube musics and extract them as embeddings</h3>
<h3>2. Implement our own ANNOY model from the embeddings : a powerful recommender used even by YouTube (stands for Approximate Nearest Neighbors Oh Yeah) - and save it</h3>
<h3>3. Implement our Flask API with ANNOY recommender and additional intents defined with Wit AI: greetings, goodbyes, thanks</h3>
<br>

Project Organization
------------

    ├── models            
    │   └── nearest_neighbor_graph.ann          <- Saved ANNOY model
    │
    ├── notebooks     
    │   └── Annoy_Recommender.ipynb             <- Implements and saves the ANNOY model
    │   └── Song Dataset Creation.ipynb         <- Creates and saves the dataset (music embeddings)
    │
    ├── src                
    │   ├── features    
    │   │   └── build_features.py               <- Extracts the genres from all musics and saves them in a dictionary
    │   │   └── Recommenders.py
    │   │   └── spotify_recommender.py          <- Back script to recommend Spotify songs (both content-based and collaborative filtering)
    │   │   └── youtube_recommender.py          <- Back script to recommend Youtube Musics from user input
    |
    ├── web_ml                
    │   ├── data       
    │   │   └── musics
    │   │   |   └── class_labels_indices.csv    <- Labels indices
    │   │   |   └── music_set.json              <- music embeddings
    │   │   |   └── flattened_labels.txt        <- Flattened YouTube labels
    │   ├── static
    │   │   └── slylesheets
    │   │   |   └── pretty-checkbox.css         <- CSS for request.html
    │   │   |   └── seasoning-style.css         <- CSS for request.html
    │   │   └── chatbot-icon.png
    │   │   └── icon.png
    │   │   └── youtube.png                     <- YoutuBot Icon
    │   ├── templates
    │   |   └── request.html                    <- Web page for our chatbot
    │   ├── app.py                              <- Flask API implementation
    │   ├── requirements.txt                    <- Requirements' file
    │   ├── Procfile                            
    │
    ├── Makefile         
    ├── .gitignore         
    ├── README.md         
    ├── requirements.txt  
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
