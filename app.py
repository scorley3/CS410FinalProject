from flask import Flask, render_template, redirect, url_for, request
from flask import Blueprint, render_template, send_from_directory
from home.home import home_blueprint
from songdata import song_info, song_meta, word_tokenize, stem_words, pd, tokenizer, mood_model, labels, pad_sequences, max_length
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gensim
from gensim.models import Word2Vec
from scipy.special import expit # sigmoid function
import numpy as np
import os 

os.environ['FLASK_DEBUG'] = 'True'

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

# transform text to tokenized form
training_data = []
for doc in song_info["lyrics"]:
    training_data.append(doc.split())
    
# train word2vec model using gensim library
modelw2v = Word2Vec(sentences=training_data, vector_size=100, window=5, min_count=1, hs=1, negative=0)

def calculate_log_likelihood(embedding_query, embedding_document):
    # compute dot product of embedding query and embedding document
    dot_product = np.dot(embedding_query, embedding_document)

    if dot_product > 0:
        # if dot prod > 0, take the log to get log likelihood
        log_likelihood = np.log(dot_product)
    else:
        # otherwise, set log_likelihood to 0 to prevent errors
        log_likelihood = 0

    # return sigmoid of the log_likelihood to normalize values to [0,1]
    return expit(log_likelihood)


def relevance_score(embedding_query, document, word2vec_model):
    # tokenize query and document
    document_tokens = word_tokenize(document)

    # get word embeddings for document
    # do this by taking the mean of all the embeddings for words in vocabulary and in document
    embedding_document = np.mean([word2vec_model.wv[word] for word in document_tokens if word in word2vec_model.wv.index_to_key], axis=0)

    # calculate log likelihood
    if not np.isnan(embedding_query).any() and np.any(embedding_query) and np.any(embedding_document):  # check if embeddings are not all zeros
        # call calculation function
        log_likelihood = calculate_log_likelihood(embedding_query, embedding_document)
    else:
        # if all embeddings are 0, set log_likelihood to default value
        log_likelihood = 0.0

    return log_likelihood

def ranking_function_w2v(query, df):
    # copy dataframe
    to_rank = df.copy()

    # get word embeddings for query
    # do this by taking the mean of all the embeddings for words in vocabulary and in query
    embedding_query = np.mean([modelw2v.wv[word] for word in word_tokenize(query) if word in modelw2v.wv.index_to_key], axis=0)

    # compute w2v relevance for each document
    for index, row in to_rank.iterrows():
        to_rank.at[index, "relevance"] = relevance_score(embedding_query, row["lyrics"], modelw2v)

    return to_rank.sort_values(by="relevance", ascending=False)

#add the 'relevance' column to the song_info dataframe
song_info['relevance'] = None

# create mood dataframe
mood_data = pd.read_csv("data_moods.csv")
md = mood_data.copy()
md.drop(columns=["name","album", "artist", "release_date", "popularity", "id", "length", "key", "time_signature"], inplace=True)

# more imports
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, confusion_matrix

def create_model():
  # initialize model
  kerasModel = Sequential()
  # add layer for data input
  kerasModel.add(Dense(8, input_dim=9, activation="relu"))
  kerasModel.add(Dense(4,activation='softmax'))
  kerasModel.compile(loss=keras.losses.SparseCategoricalCrossentropy())
  return kerasModel

# create model 
model = create_model()

# split training and testing data 
training = md.copy().iloc[:620]
testing=md.copy().iloc[620:]

# create map of moods to numbers and adjust dataframes for test and training data
mood_map = {"Happy": 0, "Sad": 1, "Energetic": 2, "Calm": 3}
input_train = training.copy().drop(columns=["mood"])
output_train = training["mood"].map(mood_map)

input_test = testing.copy().drop(columns=["mood"])
output_test = testing["mood"].map(mood_map)

# fit the model
history = model.fit(
    input_train,
    output_train,
    batch_size=5,
    epochs=50,
    validation_data=(input_test, output_test)
)

# evaluate the model
results = model.evaluate(input_test, output_test)

# predict using the model 
predictions = model.predict(input_test)

# code to check accuracy of the model on test data, uncomment to test 
#indices = np.argmax(predictions, axis=1)
# print(np.argmax(predictions, axis=1))
# correct = 0
# incorrect = 0
# for i in range(len(indices)):
#   if indices[i] == output_test.iat[i]:
#     correct += 1
#   else:
#      incorrect += 1
# print(correct, incorrect)
# print(correct / float(incorrect + correct))

# predict for song data 
data_map = song_meta.copy()
model_columns = input_train.columns.tolist()
data_pred = model.predict(data_map.copy().drop(columns=["playlist_subgenre", "key", "track_album_name", "playlist_genre", "playlist_subgenre", "duration_ms", "track_id", "track_popularity", "track_album_id", "mode"])
  .reindex(columns=model_columns))

indices = np.argmax(data_pred, axis=1)

mood_reverse_map = {0: "Happy", 1: "Sad", 2: "Energetic", 3: "Calm"}

moods = [mood_reverse_map[value] for value in indices]

# switch numerical value and mood string for data map
data_map["mood"] = moods

# make copies of the data to prevent lost of information
songs_moods_data = data_map.copy()
songs_lyrics_data = song_info.copy()

def generate_playlists(query):
  #find songs based on mood
  new_sequences = tokenizer.texts_to_sequences([query])
  updated_new_sequences = np.array(pad_sequences(new_sequences, maxlen=max_length, padding='post'))
  predictions = mood_model.predict(updated_new_sequences)
  predicted_label = [labels[prediction.argmax()] for prediction in predictions]

  mood_matching_data = songs_moods_data[songs_moods_data["mood"] == predicted_label[0]].copy()
  mood_matching_data.drop(columns=["track_popularity", "track_album_id", "track_album_name", "playlist_genre", "playlist_subgenre", "danceability", "energy", "key", "loudness", "mode", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo", "duration_ms"], inplace=True)

  # find songs based on lyrics and query
  stem_query = stem_words(query)
  lyrics_matching_data = ranking_function_w2v(stem_query, songs_lyrics_data).drop(columns=["lyrics"], inplace=False)

  # merge data frames for lyric and mood matches 
  top_songs_moods = pd.merge(mood_matching_data, lyrics_matching_data, on="track_id", how='inner')
  top_songs_moods = top_songs_moods.sort_values(by="relevance", ascending=False)

  return top_songs_moods



@app.route('/')
def home():
    return render_template('home.html')

@app.route('/generate', methods=['GET'])
def load(): 
    # retrieve query from request
    query = request.args.get("phrase")
    
    # code to get generate playlist
    rel_songs = generate_playlists(query)

    # return rendering of output page with list of songs 
    return render_template('index.html', songs=rel_songs.iloc[:50], query=query)


if __name__ == "__main__":
    app.run(debug=True)