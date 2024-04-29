import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
import numpy as np

nltk.download("stopwords")
nltk.download("punkt")

def parse_corpus(file):
  df = pd.read_csv(file)
  df = df.iloc[:3000] #uncomment for testing purposes
  cols_to_drop = ["track_album_release_date", "playlist_name", "playlist_id", "language"]
  df.drop(columns=cols_to_drop, inplace=True)

  #some songs dont have lyrics, so we need to remove songs
  df = df[df["lyrics"].apply(lambda x: isinstance(x, str))]

  # make two separate data frames
  # 1. df_info: holds general song information (title, lyrics)
  # 2. df_meta: holds more specific song information (duration, bpm, etc.)
  # both have the track_id for easy access between the two

  #probably not neccessay, but for organization. I didnt want to store metadata with lyrics if its not being manipulated

  df_info = df.copy()
  df_info = df.loc[:, "track_id":"lyrics"]

  df_meta = df.copy()
  cols_to_drop = ["track_name", "track_artist", "lyrics"]
  df_meta.drop(columns=cols_to_drop, inplace=True)

  return df_info, df_meta

def parse_corpus_keep_playlist(file):
  df = pd.read_csv(file)
  # df = df.iloc[:3000] #uncomment for testing purposes
  cols_to_drop = ["track_album_release_date", "language"]
  df.drop(columns=cols_to_drop, inplace=True)

  #some songs dont have lyrics, so we need to remove songs
  df = df[df["lyrics"].apply(lambda x: isinstance(x, str))]

  # make two separate data frames
  # 1. df_info: holds general song information (title, lyrics)
  # 2. df_meta: holds more specific song information (duration, bpm, etc.)
  # both have the track_id for easy access between the two

  #probably not neccessay, but for organization. I didnt want to store metadata with lyrics if its not being manipulated

  df_info = df.copy()
  df_info = df.loc[:, "track_id":"lyrics"]

  df_meta = df.copy()
  cols_to_drop = ["track_name", "track_artist", "lyrics"]
  df_meta.drop(columns=cols_to_drop, inplace=True)

  return df_info, df_meta

song_info, song_meta = parse_corpus("spotify_songs.csv")

# functions used to remove stuff from text
def remove_punctuation(text):
    text = re.sub(r'[-_._\\]', ' ', text)
    text = re.sub(r'\'s\b', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

def remove_white_space(text):
    text = ' '.join(text.split())
    return text

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

def stem_words(text):
    words = word_tokenize(text)
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in words]
    stemmed_text = ' '.join(stemmed_words)
    return stemmed_text

def parse_text(df):

    #transform words to lower case
    df['lyrics'] = df['lyrics'].str.lower()

    #remove numbers
    df['lyrics'] = df['lyrics'].str.replace('\d+', '')

    #remove punctunation
    df['lyrics'] = df['lyrics'].apply(remove_punctuation)

    #remove excess white space
    df['lyrics'] = df['lyrics'].apply(remove_white_space)

    #remove stop words
    df['lyrics'] = df['lyrics'].apply(remove_stopwords)

    #stem words in the lyrics
    df['lyrics'] = df['lyrics'].apply(stem_words)

    #remove rows where entire lyrics are cleared
    df = df[df["lyrics"] != '']

    #get the top 500 frequent lyrics
    # text = ' '.join(df['lyrics'])
    # words = text.split()
    # word_counts = Counter(words)
    # top_500_words = word_counts.most_common(500)
    # top_words = []
    # for val in top_500_words:
    #     top_words.append(val[0])

    return df# top_words

song_info = parse_text(song_info)
print(song_info)


# import necessary libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def process_emotions(mood):
  if mood == "joy":
    return "Happy"
  elif mood == "sadness" or  mood == "fear":
    return "Sad"
  elif mood == "anger" or mood == "surprise":
    return "Energetic"
  elif mood == "love":
    return "Calm"

# parse the csv file to split the sentence and its corresponding mood
sentence_moods = pd.read_csv("sentence_mood.txt")

split_data = sentence_moods["words"].str.split(';', expand=True)
if len(split_data.columns) == 2:
  sentence_moods[['sentence', 'emotion']] = split_data

# redefine emotions to the defined ones
sentence_moods['emotion'] = sentence_moods['emotion'].apply(process_emotions)
# app.register_blueprint(home_blueprint)

# define texts and labels to train model
texts = list(sentence_moods["sentence"])
labels = list(sentence_moods["emotion"])

# tokenize text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

# define numerical sequence
sequences = tokenizer.texts_to_sequences(texts)

# pad sequences to ensure equal length for all sequences
max_length = max([len(seq) for seq in sequences])
updated_sequences = np.array(pad_sequences(sequences, maxlen=max_length, padding='post'))

# convert labels to numerical representations - 0 to 3 (for the four moods)
label_indices = np.array([labels.index(label) for label in labels])


# define model architecture
mood_model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=16, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(4, activation='softmax')
])

# compile the model
mood_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# train the model
mood_model.fit(updated_sequences, label_indices, epochs=75, verbose=1)