from flask import Flask, render_template, redirect, url_for, request
from flask import Blueprint, render_template, send_from_directory
from home.home import home_blueprint
from songdata import song_info, song_meta, word_tokenize, stem_words, pd
import os 

os.environ['FLASK_DEBUG'] = 'True'

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

# functions used to remove stuff from text

import gensim
from gensim.models import Word2Vec
from scipy.special import expit # sigmoid function
import numpy as np

print("in main file!!!!!")
print(song_info)
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

# function to compute word relevances using a vector space representation using word2vec
def relevance_score(query, document, word2vec_model):
    # tokenize query and document
    query_tokens = word_tokenize(query)
    document_tokens = word_tokenize(document)

    # get word embeddings for query and document
    # do this by taking the mean of all the embeddings for words in vocabulary and in query or document, respectively
    # embedding_query = np.mean([word2vec_model.wv[word] for word in query_tokens if word in word2vec_model.wv.index_to_key], axis=0)
    # embedding_document = np.mean([word2vec_model.wv[word] for word in document_tokens if word in word2vec_model.wv.index_to_key], axis=0)
    embedding_query = np.mean([word2vec_model.wv[word] for word in query_tokens if word in word2vec_model.wv.index_to_key], axis=0)
    embedding_document = np.mean([word2vec_model.wv[word] for word in document_tokens if word in word2vec_model.wv.index_to_key], axis=0)

    # calculate log likelihood
    if np.any(embedding_query) and np.any(embedding_document):  # check if embeddings are not all zeros
        # call calculation function
        log_likelihood = calculate_log_likelihood(embedding_query, embedding_document)
    else:
        # if all embeddings are 0, set log_likelihood to default value
        log_likelihood = 0.0

    return log_likelihood

def ranking_function_w2v(query, df):
    # copy dataframe
    to_rank = df.copy()

    # compute w2v relevance for each document
    for index, row in to_rank.iterrows():
        to_rank.at[index, "relevance"] = relevance_score(query, row["lyrics"], modelw2v)

    # print(to_rank)
    # drop class and title columns for readable output
    # to_rank = to_rank.drop(columns=['class', 'title'])

    # return dataframe sorted by descending relevance
    return to_rank.sort_values(by="relevance", ascending=False)

#add the 'relevance' column to the song_info dataframe
song_info['relevance'] = None

# print(ranking_function_w2v(stem_words("happy shopping dancing"), song_info))
# testing model
# print("Document Relevances for Test Queries")
# queries = ["happy shopping dancing", "majestic confident walk", "spooky mysterious night", "hopeless romantic", "lyrical poetic", "soul-crushing lonely"]
# for query in queries:
#   stem_query = stem_words(query)
#   print(f"1. {query}: \n", ranking_function_w2v(stem_query, song_info))

mood_data = pd.read_csv("data_moods.csv")
md = mood_data.copy()
md.drop(columns=["name","album", "artist", "release_date", "popularity", "id", "length", "key", "time_signature"], inplace=True)
# print(md)

import keras
from keras.models import Sequential
from keras.layers import Dense
#from keras.wrappers.scikit_learn import KerasClassifier
#from keras.utils import np_utils
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

model = create_model()
training = md.copy().iloc[:620]
testing=md.copy().iloc[620:]
mood_map = {"Happy": 0, "Sad": 1, "Energetic": 2, "Calm": 3}
input_train = training.copy().drop(columns=["mood"])
output_train = training["mood"].map(mood_map)

input_test = testing.copy().drop(columns=["mood"])
output_test = testing["mood"].map(mood_map)

history = model.fit(
    input_train,
    output_train,
    batch_size=5,
    epochs=50,
    validation_data=(input_test, output_test)
)

# print(history.history)

results = model.evaluate(input_test, output_test)
# print(results)

predictions = model.predict(input_test)
# print("predictions shape:", predictions.shape)
# #print(predictions)
# print(testing)
# print(output_test)
indices = np.argmax(predictions, axis=1)
# print(np.argmax(predictions, axis=1))
correct = 0
incorrect = 0
# for i in range(len(indices)):
#   if indices[i] == output_test.iat[i]:
#     correct += 1
#   else:
#      incorrect += 1
# print(correct, incorrect)
# print(correct / float(incorrect + correct))

data_map = song_meta.copy()
model_columns = input_train.columns.tolist()
data_pred = model.predict(data_map.copy().drop(columns=["playlist_subgenre", "key", "track_album_name", "playlist_genre", "playlist_subgenre", "duration_ms", "track_id", "track_popularity", "track_album_id", "mode"])
  .reindex(columns=model_columns))
# print(data_pred)
indices = np.argmax(data_pred, axis=1)
# print(indices)
mood_reverse_map = {0: "Happy", 1: "Sad", 2: "Energetic", 3: "Calm"}

moods = [mood_reverse_map[value] for value in indices]

data_map["mood"] = moods

# print(data_map)
print(data_map["mood"].value_counts())

# app.register_blueprint(home_blueprint)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/generate', methods=['GET'])
def load(): 
    query = request.args.get("phrase")
    # code to get query mood 
    
    # code to get query word2vec 
    
    # ranking 
    rel_songs = ranking_function_w2v(stem_words(query), song_info)
    return render_template('index.html', songs=rel_songs)

if __name__ == "__main__":
    app.run(debug=True)