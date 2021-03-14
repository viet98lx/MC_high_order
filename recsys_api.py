from flask import Flask, request, render_template, jsonify
from elasticsearch import Elasticsearch
import scipy.sparse as sp
import MC_utils
import glob
from MC import MarkovChain
from time import sleep, time
import json
import os
import requests

es = Elasticsearch(hosts='localhost:9200')
app = Flask(__name__)

@app.route('/analytics/seqrec')
def recommend_list_item(transaction_df, model_path):
    transaction_df.sort_values(['user_id', 'date'])
    groupby_date_df = transaction_df.groupby(['user_id', 'date'])['item_id'].apply(list).reset_index(name='list_item')
    bseq = []
    for i in range(len(groupby_date_df)):
        list_item = groupby_date_df.loc[i, "list_item"]
        bseq.append(list_item)
    trans_matrix_path = glob.glob("*.npz")
    list_trans_matrix = []
    for trans_matrix_path in trans_matrix_path:
        list_trans_matrix.append(sp.load_npz(trans_matrix_path))
    mc_order = len(list_trans_matrix)

    item_dict_path = model_path+'/item_dict.json'
    with open(item_dict_path, 'r') as fp:
        item_dict = json.load(fp)

    item_freq_dict_path = model_path + '/item_freq_dict.json'
    with open(item_freq_dict_path, 'r') as fp:
        item_freq_dict = json.load(fp)

    reversed_item_dict_path = model_path + '/reversed_item_dict.json'
    with open(reversed_item_dict_path, 'r') as fp:
        reversed_item_dict = json.load(fp)

    w_behavior_file = model_path + '/w_behavior.json'
    with open(w_behavior_file, 'r') as fp:
        w_behavior = json.load(fp)
    mc_model = MarkovChain(item_dict, reversed_item_dict, item_freq_dict, w_behavior, list_trans_matrix, mc_order)
    previous_basket = []
    for item in bseq[-1]:
        if item in item_dict:
            previous_basket += item
    topk = 10
    try:
        list_recommend = mc_model.top_predicted_item(previous_basket, topk)
    except:
        popular_dict = dict(sorted(item_freq_dict.items(), key=lambda item: item[1], reverse=True))
        list_recommend = popular_dict.keys()[:topk]
    return list_recommend

if __name__ == '__main__':
    app.run(debug=True, port=5000)