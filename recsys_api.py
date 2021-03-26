from flask import Flask, request, render_template, jsonify
import pandas as pd
import json
from datetime import date
import pickle
import heapq
import glob

app = Flask(__name__)
# csv_file = glob.glob('data/*.csv')[0]
# csv_file = 'data/data.csv'
# transaction_df = pd.read_csv(csv_file)

def load_model(model_file_path):
    with open(model_file_path, 'rb') as input:
        restored_model = pickle.load(input)
    return restored_model

def recommend(mc_model, bseq, topk):
    previous_baskets = []
    item_dict = mc_model.item_dict
    item_freq_dict = mc_model.item_freq_dict
    try:
        last_basket = []
        for item in bseq[-1]:
            if item in item_dict:
                last_basket.append(item)
        if len(last_basket) >= 1:
            previous_baskets.append(last_basket)
        # print(previous_baskets)
        if len(previous_baskets) < 1:
            # print(previous_baskets)
            raise Exception("Empty basket")
        list_recommend, list_score = mc_model.top_predicted_mc_order_with_score(previous_baskets, topk)
        list_rank = [i for i in range(len(list_recommend), 0, -1)]
        list_result = []
        for j in range(len(list_recommend)-1, -1, -1):
            tup = (list_recommend[j], list_score[j], list_rank[j])
            list_result.append(tup)
    except:
        # popular_dict = dict(sorted(mc_model.item_freq_dict.items(), key=lambda item: item[1], reverse=True))
        list_recommend = heapq.nlargest(topk, item_freq_dict, key=item_freq_dict.get)
        # list_recommend = list(popular_dict.keys())[:topk]
        sum_freq = sum(list(item_freq_dict.values()))
        list_score = [item_freq_dict[item]/sum_freq for item in list_recommend]
        list_rank = [i for i in range(1, len(list_recommend) + 1)]
        list_result = []
        for j in range(0, len(list_recommend)):
            tup = (list_recommend[j], list_score[j], list_rank[j])
            list_result.append(tup)

    return list_result

def filter_data(task, data_df):
    list_item_groupby_date_df = data_df.groupby(['user_id', 'date'])['item_id'].apply(list).reset_index(name='list_item')
    list_behavior_groupby_date_df = data_df.groupby(['date'])['behavior'].apply(list).reset_index(name='list_behavior')
    print(list_behavior_groupby_date_df)
    bseq = []
    for i in range(len(list_item_groupby_date_df)):
        list_item = list_item_groupby_date_df.loc[i, "list_item"]
        list_behavior = list_behavior_groupby_date_df.loc[i, "list_behavior"]
        list_pair = []
        for i, b in zip(list_item, list_behavior):
            print(b)
            if task == 1 and (b == 'cart' or b == 'buy'):
                list_pair += [i]
            if task == 2 and b != 'buy':
                list_pair += [i]
            if task == 3 and b == 'pv':
                list_pair += [i]
        bseq.append(list_pair)
    return bseq

@app.route('/analytics/seqrec', methods=['POST'])
def recommend_list_item():
    # customer_id = int(request.args.get('customer_id'))
    # json_data = request.json
    json_data = request.get_json()
    customer_id = json_data["customer_id"]
    # print(type(customer_id))
    # user_seq_df = transaction_df[transaction_df["user_id"] == customer_id]
    # sort_user_seq_df = user_seq_df.sort_values(['user_id', 'date'])
    try:
        # last_basket = request.args.getlist('last_basket')
        last_basket = json_data["last_basket"]
        str_last_basket = [str(item_id) for item_id in last_basket]
    except:
        response = {"items": [],
                    "time": date.today(),
                    "message": "Last basket is empty",
                    "status": 200}
        return jsonify(response)

    # task = int(request.args.get('task'))
    task = json_data["task"]
    # topk = int(request.args.get('topk'))
    topk = json_data["topk"]
    print(type(task))
    if task == 1: # recommend buy target
        print("Recommend buy")
        model_folder = 'model/mc_model/buy_model.pkl'
    elif task == 2: # recommend cart target
        print("Recommend cart")
        model_folder = 'model/mc_model/cart_model.pkl'
        # mc_model = load_model(model_folder)
    elif task == 3: # recommend pv target
        print("Recommend pv")
        model_folder = 'model/mc_model/pv_model.pkl'
    else:
        response = {"items": [],
                    "time": date.today(),
                    "message": "Don't support this task",
                    "status": 200}
        return jsonify(response)

        # mc_model = load_model(model_folder)
    # bseq = filter_data(task, sort_user_seq_df)
    bseq = [str_last_basket]
    mc_model = load_model(model_folder)
    list_result = recommend(mc_model, bseq, topk)
    response = {"items" : list_result,
                "time" : date.today(),
                "message": "recommend {} items".format(topk),
                "status": 200}
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port=5000)