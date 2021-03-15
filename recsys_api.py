from flask import Flask, request, render_template, jsonify
from elasticsearch import Elasticsearch
import scipy.sparse as sp
import MC_utils
import glob
from MC import MarkovChain
import pandas as pd
from time import sleep, time
import json
from datetime import date
import os
import requests

es = Elasticsearch(hosts='localhost:9200')
app = Flask(__name__)
transaction_df = pd.read_csv('data/seqrec_sample_large.csv', nrows=150000)

def load_model(model_folder):
    trans_matrix_path = glob.glob(model_folder + "*.npz")
    list_trans_matrix = []
    for trans_matrix_path in trans_matrix_path:
        list_trans_matrix.append(sp.load_npz(trans_matrix_path))
    mc_order = len(list_trans_matrix)

    item_dict_path = model_folder + '/item_dict.json'
    with open(item_dict_path, 'r') as fp:
        item_dict = json.load(fp)

    item_freq_dict_path = model_folder + '/item_freq_dict.json'
    with open(item_freq_dict_path, 'r') as fp:
        item_freq_dict = json.load(fp)

    reversed_item_dict_path = model_folder + '/reversed_item_dict.json'
    with open(reversed_item_dict_path, 'r') as fp:
        reversed_item_dict = json.load(fp)

    w_behavior_file = model_folder + '/w_behavior.json'
    with open(w_behavior_file, 'r') as fp:
        w_behavior = json.load(fp)
    mc_model = MarkovChain(item_dict, reversed_item_dict, item_freq_dict, w_behavior, list_trans_matrix, mc_order)
    return mc_model

def recommend(mc_model, bseq):
    previous_basket = []
    for basket in bseq:
        for item in basket:
            if item in mc_model.item_dict:
                previous_basket += item
    topk = 10
    print("len of last basket", len(previous_basket))
    try:
        list_result = mc_model.top_predicted_item(previous_basket, topk)
    except:
        popular_dict = dict(sorted(mc_model.item_freq_dict.items(), key=lambda item: item[1], reverse=True))
        list_recommend = list(popular_dict.keys())[:topk]
        list_score = [popular_dict[item] for item in list_recommend]
        list_rank = [i for i in range(1, len(list_recommend) + 1)]
        list_result = []
        for j in range(0, len(list_recommend)):
            tup = (list_recommend[j], list_score[j], list_rank[j])
            list_result.append(tup)

    return list_result
def filter_data(task, data_df):
    list_item_groupby_date_df = data_df.groupby(['user_id', 'date'])['item_id'].apply(list).reset_index(name='list_item')
    list_behavior_groupby_date_df = data_df.groupby(['user_id', 'date'])['behavior'].apply(list).reset_index(name='list_behavior')
    bseq = []
    for i in range(len(list_item_groupby_date_df)):
        list_item = list_item_groupby_date_df.loc[i, "list_item"]
        list_behavior = list_behavior_groupby_date_df.loc[i, "list_behavior"]
        list_pair = []
        for i, b in zip(list_item, list_behavior):
            if task == 1 and (b == 'cart' or b == 'buy'):
                list_pair += [i]
            if task == 2 and b != 'buy':
                list_pair += [i]
            if task == 3 and b == 'pv':
                list_pair += [i]
        bseq.append(list_pair)
    return bseq

@app.route('/analytics/seqrec')
def recommend_list_item():
    customer_id = int(request.args.get('customer_id'))
    print(type(customer_id))
    user_seq_df = transaction_df[transaction_df["user_id"] == customer_id]
    sort_user_seq_df = user_seq_df.sort_values(['user_id', 'date'])

    task = int(request.args.get('task'))
    print(type(task))
    if task == 1: # recommend buy target
        print("Recommend buy")
        model_folder = 'model/buy_target/'
    elif task == 2: # recommend cart target
        print("Recommend cart")
        model_folder = 'model/cart_target/'
        # mc_model = load_model(model_folder)
    elif task == 3: # recommend pv target
        print("Recommend pv")
        model_folder = 'model/pv_target/'

        # mc_model = load_model(model_folder)
    bseq = filter_data(task, sort_user_seq_df)
    mc_model = load_model(model_folder)
    list_result = recommend(mc_model, bseq)
    response = {"items" : list_result,
                "time" : date.today(),
                "message": "recommend most frequent item",
                "status": 200}
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port=5000)