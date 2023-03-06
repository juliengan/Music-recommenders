YoutuBot
==============================

Chatbot that recommends musics from YouTube 

Project Organization
------------

    ├── LICENSE
    ├── Makefile         
    ├── README.md         
    ├── models             <- Trained and serialized ANN model
    │
    ├── notebooks          
    │
    ├── requirements.txt  
    │
    ├── src                
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │   └── __init__.py
    │   │   └── Recommenders.py
    │   │   └── spotify_recommender.py
    │   │   └── youtube_recommender.py
    ├── web_ml                
    │   ├── data       
    │   ├── static
    │   ├── templates
    │   |   └── recommend.html
    │   |   └── request.html
    │   ├── app.py
    │   ├── requirements.txt
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
