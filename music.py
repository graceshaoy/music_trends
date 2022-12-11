import pandas as pd
import nltk
import string
from gensim import corpora, models
from gensim.utils import effective_n_jobs
import collections
import re
import matplotlib.pyplot as plt
import numpy as np
from bs4 import BeautifulSoup
import access
import networkx as nx
import importlib
import ndlib
importlib.reload(ndlib)
import ndlib.models.CompositeModel as gc
from ndlib.models.compartments import NodeThreshold
from ndlib.models.compartments import NodeStochastic
from ndlib.utils import multi_runs
import ndlib.models.ModelConfig as mc
from requests.exceptions import Timeout

# import spotipy
# from spotipy.oauth2 import SpotifyClientCredentials
# import requests
# client_credentials_manager = SpotifyClientCredentials(client_id = access.spotify_client_id, client_secret = access.spotify_client_secret)
# sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)

# import lyricsgenius
# from rauth import OAuth2Service

STOP = set(nltk.corpus.stopwords.words('english') + list(string.punctuation))

def get_wordnet_pos(word):
    '''
    Tags each word with its Part-of-speech indicator -- specifically used for
    lemmatization in the get_lemmas function
    '''
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {'J': nltk.corpus.wordnet.ADJ,
                'N': nltk.corpus.wordnet.NOUN,
                'V': nltk.corpus.wordnet.VERB,
                'R': nltk.corpus.wordnet.ADV}

    return tag_dict.get(tag, nltk.corpus.wordnet.NOUN)

def get_lemmas(text):
    '''
    Gets lemmas for a string input, excluding stop words, punctuation, as well
    as a set of study-specific stop-words
    '''
    lemmas = [nltk.stem.WordNetLemmatizer().lemmatize(t, get_wordnet_pos(t))
              for t in nltk.word_tokenize(text.lower()) if t not in STOP]
    return lemmas

def make_bigrams(lemmas):
    '''
    Make bigrams for words within a given document
    '''
    bigram = models.Phrases(lemmas, min_count=5)
    bigram_mod = bigram.freeze()
    return [bigram_mod[doc] for doc in lemmas]

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=2):
    '''
    Computes Coherence values for LDA models with differing numbers of topics.

    Returns list of models along with their respective coherence values (pick
    models with the highest coherence)
    '''
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = models.ldamulticore.LdaMulticore(corpus=corpus,
                                                 id2word=dictionary,
                                                 num_topics=num_topics,
                                                 workers=effective_n_jobs(-1))
        model_list.append(model)
        coherence_model = models.coherencemodel.CoherenceModel(model=model,
                                                          corpus=corpus,
                                                          dictionary=dictionary,
                                                          coherence='u_mass')
        coherence_values.append(coherence_model.get_coherence())

    return model_list, coherence_values

def fill_topic_weights(df_row, bow_corpus, ldamodel):
    '''
    Fill DataFrame rows with topic weights for topics in songs.

    Modifies DataFrame rows *in place*.
    '''
    try:
        for i in ldamodel[bow_corpus[df_row.name]]:
            df_row[str(i[0])] = i[1]
    except:
        return df_row
    return df_row

def set_legend_alpha(leghandles, alpha = 1):
    for l in leghandles:
        l.set_alpha(alpha)

def plot_numerical_std(var_group,all_genres, by_genre, legend = False):
    i = 1
    plt.figure(figsize=(16,8))
    for var in var_group:
        if i == 1:
            ax1 = plt.subplot(2,5,i)
        else:
            plt.subplot(2,5,i,sharey=ax1)
        all = pd.DataFrame()
        all['year'] = np.arange(1950,2016)
        for genre in all_genres:
            df = by_genre[genre].copy()
            df = df.groupby('year').std().reset_index()
            plt.plot(df['year'], df[var].rolling(window=10).mean(),label=genre, alpha = 0.25)
            all = all.merge(df[['year',var]], on='year', how='left')
        all['avg_std'] = all[all.columns[1:]].mean(axis=1)
        plt.plot(all['year'], all['avg_std'].rolling(window=10).mean(),label='avg std of genres', c='black')
        plt.ylabel('std')
        plt.title(var)
        i += 1
    if legend:
        leg = plt.legend(labels = all_genres, bbox_to_anchor=(1.05,1),ncol=2)
        set_legend_alpha(leg.legendHandles)
        # for l in leg.legendHandles:
        #     l.set_alpha(1)
        leg

# r = requests.post('https://accounts.spotify.com/api/token',
# data ={'grant_type': 'client_credentials'},auth=(access.spotify_client_id, access.spotify_client_secret))
# bearer = dict(r.json())['access_token']

def spotify_query(genre,year,max=10000):
    uri,artist_name,track_name,popularity,track_id,release_date = [],[],[],[],[],[]
    for i in range(0,max,50):
        try: 
            track_res = sp.search(q=f'genre:{genre} year:{year}', type='track', limit=50, offset=i)
            for i, t in enumerate(track_res['tracks']['items']):
                uri.append(t['uri'])
                artist_name.append(t['artists'][0]['name'])
                track_name.append(t['name'])
                popularity.append(t['popularity'])
                track_id.append(t['id'])
                release_date.append(t['album']['release_date'])
        except:
            # print(i)
            break
    df = pd.DataFrame({'uri':uri,'artist_name':artist_name, 'track_name':track_name, 'popularity':popularity, 'track_id':track_id, 'release_date':release_date})
    df['release_date'] = pd.to_datetime(df['release_date'])
    df['year'] = df['release_date'].dt.year
    df = df.set_index('uri')
    return df

def get_audio_features(df):
    features = []
    if len(df) != 0:
        for i in range(0, len(df), 50):
            try:
                features.extend(sp.audio_features(df.index[i:i+50]))
            except:
                print(i)
                break
        features = pd.DataFrame(features).set_index('uri')
        df_features = pd.merge(df, features, left_index=True, right_index=True)
        return df_features

# function to scrape lyrics from genius
# from "How to Leverage Spotify API + Genuis Lyrics for Data Science Tasks in Python" by Maaz Khan - https://medium.com/swlh/how-to-leverage-spotify-api-genius-lyrics-for-data-science-tasks-in-python-c36cdfb55cf3
def scrape_lyrics(artistname, songname):
    artistname2 = str(artistname.replace(' ','-')) if ' ' in artistname else str(artistname)
    songname2 = str(songname.replace(' ','-')) if ' ' in songname else str(songname)
    page = requests.get('https://genius.com/'+ artistname2 + '-' + songname2 + '-' + 'lyrics')
    html = BeautifulSoup(page.text, 'html.parser')
    lyrics1 = html.find("div", class_="lyrics")
    lyrics2 = html.find("div", class_="Lyrics__Container-sc-1ynbvzw-2 jgQsqn")
    if lyrics1:
        lyrics = lyrics1.get_text()
    elif lyrics2:
        lyrics = lyrics2.get_text()
    elif lyrics1 == lyrics2 == None:
        lyrics = None
    return lyrics

# function to attach lyrics onto data frame - edited from Maaz Khan's code
def get_lyrics(df):
    for i,x in enumerate(zip(df['artist_name'],df['track_name'])):
        test = scrape_lyrics(x[0], x[1])
        df.loc[i, 'lyrics'] = test
    return df

# Function returns results from all simulated runs for plotting 95% CIs
def full_simulate_net_diffusion(frac_infected=0.01 ,threshold=0.038,
                                profile=0.0000105, p_removal=0.22,
                                num_exec=20, num_iter=32, nproc=8):
    # Network generation
    g = nx.erdos_renyi_graph(1000, 0.1)
    
    # Composite Model instantiation
    sir_th_model = gc.CompositeModel(g)

    # Model statuses
    sir_th_model.add_status("Susceptible")
    sir_th_model.add_status("Infected")
    sir_th_model.add_status("Removed")

    # Compartment definition
    c1 = NodeThreshold(threshold=None, triggering_status="Infected")
    c2 = NodeStochastic(p_removal)

    # Rule definition
    sir_th_model.add_rule("Susceptible", "Infected", c1)
    sir_th_model.add_rule("Infected", "Removed", c2)

    # Model initial status configuration, assume 1% of population is infected
    config = mc.Configuration()
    config.add_model_parameter('fraction_infected', frac_infected)

    # Setting nodes parameters
    for i in g.nodes():
        config.add_node_configuration("threshold", i, threshold)
        config.add_node_configuration("profile", i, profile)

    # Simulation execution
    sir_th_model.set_initial_status(config)
    trends = multi_runs(sir_th_model, execution_number=num_exec,
                        iteration_number=num_iter, nprocesses=nproc)
    
    # Convert into a dataframe that lists each number of infected nodes by
    # iteration number (to make average calculation)
    df_infected = pd.DataFrame([execution['trends']['node_count'][1]
                                for execution in trends])
    
    # Scale each run (0-100), so that they're consistent with Google Trend Data
    # for comparison:
    df_infected = df_infected.apply(lambda x: x/x.max(), axis=1)
    df_infected = pd.melt(df_infected,
                          var_name='Execution',
                          value_name='Infected')
    df_infected['Infected'] *= 100
    
    return df_infected

def get_lyrics(df):
    lyrics = []
    genius = OAuth2Service(client_id=access.genius_client_id, client_secret=access.genius_client_secret, name='genius', authorize_url='https://api.genius.com/oauth/authorize', access_token_url='https://api.genius.com/oauth/token', base_url='https://api.genius.com/')
    genius = lyricsgenius.Genius(access.genius_access_token)
    genius.verbose = False
    genius.remove_section_headers = True
    genius.timeout = 15
    genius.sleep_time = 40
    for track in df.values:
        try:
            song = genius.search_song(track[0], track[1])
        except Timeout as e:
            continue
        if song is not None:
            lyrics.append(song.lyrics)
        else:
            lyrics.append(np.NAN)
    df['lyrics'] = lyrics
    return df