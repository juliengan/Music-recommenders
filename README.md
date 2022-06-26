# Music Recommender using Annoy 

An audio-based recommendation system from scratch using Annoy algorithm.

## Process

1. I already downloaded the audio Embedding dataset (YouTube videos encoded with [MAX-Audio-Embedding-Generator](https://github.com/IBM/MAX-Audio-Embedding-Generator)) from [AudioSet](https://research.google.com/audioset/download.html). The file is : `storage.googleapis.com/eu_audioset/youtube_corpus/v1/features/features.tar.gz` and has been decompressed.

2. Run `Annoy Recommender.ipynb` which uses a nearest neighbor algorithm to take an input audio sample and return a list of recommendations within our dataset.

3. Test through the web site : python -m flask run and go to localhost:5000