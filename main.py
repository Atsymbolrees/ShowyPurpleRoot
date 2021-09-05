import json
import urllib
import time
import logging
import uuid

#pythonsocket library for socket communication
import socket

#socket library for http communication
from http.client import HTTPRequest, HTTPResponse

#tensorflow library for natural language processing
from tensorflow.keras.preprocessing import Tokenizer
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dropout, Dense, LSTM, Activation, Embedding
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import callbacks, applications, backend

#for use of tokenizer
from nltk.tokenize import word_tokenize

#logging
import logging

#logging to the console
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    datefmt='%m-%d-%Y %H:%M:%S', level=logging.DEBUG)

#set up your environment
app.config['MONGO_DB'] = "mongodb://localhost/chatbot"
app.config['KEY'] = "KEY"
app.config['SECRET'] = "SECRET"
app.config['REFRESH_INTERVAL'] = 3600
app.config['CRAWLER'] = "/Users/rees/work/chatbot/data/scrapers/"
app.config['CHAT_PORT'] = 5000
app.config['CHAT_PORT'] = "5000"

model = ChatbotModel()
#load saved model
model.load("./models/save/chatbot_model.h5")

logging.info("Loading model saved")

#init of mongodb
from pymongo import MongoClient

#init of mongo
client = MongoClient(app.config['MONGO_DB'])
db = client.Chatbot_dataset

#init the data to store
collection = db.chatbot_dataset

def clean_text(text):
    """Return a sentence with any of the following "contaminated" strings
    removed: punctuation, stopwords, emojis."""

    from nltk.corpus import stopwords

    return text.translate(str.maketrans("","",string.punctuation))

def get_dataset_sentences(data_id, dataset):
    """Get data sentence given its id and dataset.
    Params:
    data_id: id of a data document in a dataset
    dataset: id of a dataset from which to fetch data documents"""
    log.info(dataset)
    query = {"_id":data_id}
    result = db.get_data_document(query)
    sentences = []
    for document in result.iter_documents():
        #print(document.data)
        sentences.append(clean_text(document.data))

    return sentences

def get_dataset_sentence(data_id, dataset):
       """Get a single data sentence given its id and dataset."""
    log.info(dataset)
    sentence = db.get_data_document_sentence(data_id)
    return clean_text(sentence.data)

def delete_sentence_from_db(data_id, sentence, dataset):
    """Delete a sentence from database given it's id and sentence."""
    log.info(dataset)
    # if text already removed - then sentence is already deleted.
    if sentence in db.sentences_removed:
        return

    # check if we are allowed to delete it.
    if not db.check_deletion_permission(sentence, data_id):
        log.info("Sentence not allowed to be deleted.")
        return
    log.info("Deleting sentence '{}' from data '{}'.".format(sentence, data_id))

    db.delete_data_sentence(data_id, sentence)

def count_dataset_entries(dataset):
    count = 0
    for data in dataset:
        count += 1

    return count

def print_all_dataset_sentences(dataset, data_type):
    """Print all dataset entries given dataset and data_type."""
    for entry in dataset:
        if data_type == "raw_corpus":
                       print entry["data"]
        elif data_type == "transcribed":
            print entry["data"], "--->", entry["data_id"]
            for sentence in entry["sentences_in_transcription"]:
                print sentence
        else:
            print entry["data"], "--->", entry["data_id"], "--->", entry["data_id"]
            for sentence in entry["sentences_removed"]:
                print sentence

def count_dataset_entries_by_name(dataset_name,data_type="raw_corpus"):
    dataset = db.Datasets.find_one({"name":dataset_name})
    if dataset:
        return dataset.count(data_type)
    else:
        return 0