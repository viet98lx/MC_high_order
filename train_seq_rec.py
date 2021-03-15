import os, glob
import pandas as pd
import re
import sys
import numpy as np
import MC_utils
from MC import MarkovChain
import argparse
import scipy.sparse as sp
import os
import json

def train_model(train_instances, model_name, o_dir):
    w_behavior = {'buy': 1, 'cart': 1, 'fav': 1, 'pv': 1}
    print("---------------------@Build knowledge-------------------------------")
    MAX_SEQ_LENGTH, item_dict, reversed_item_dict, item_probs, item_freq_dict, user_dict = MC_utils.build_knowledge(
        train_instances, w_behavior)
    print('Build knowledge done')
    mc_order = 1
    list_transition_matrix = MC_utils.calculate_transition_matrix(train_instances, item_dict, item_freq_dict,
                                                                  reversed_item_dict, w_behavior, mc_order)

    if not os.path.exists(o_dir):
        os.makedirs(o_dir)

    item_dict_file = os.path.join(o_dir, 'item_dict.json')
    with open(item_dict_file, 'w') as fp:
        json.dump(item_dict, fp)

    reversed_item_dict_file = os.path.join(o_dir, 'reversed_item_dict.json')
    with open(reversed_item_dict_file, 'w') as fp:
        json.dump(reversed_item_dict, fp)

    item_freq_dict_file = os.path.join(o_dir, 'item_freq_dict.json')
    with open(item_freq_dict_file, 'w') as fp:
        json.dump(item_freq_dict, fp)

    w_behavior_path = os.path.join(o_dir, 'w_behavior.json')
    with open(w_behavior_path, 'w') as fp:
        json.dump(w_behavior, fp)

    for i in range(len(list_transition_matrix)):
        sp_matrix_path = model_name + '_transition_matrix_MC_' + str(i + 1) + '.npz'
        # nb_item = len(item_dict)
        # print('Density : %.6f' % (transition_matrix.nnz * 1.0 / nb_item / nb_item))
        saved_file = os.path.join(o_dir, sp_matrix_path)
        print("Save model in ", saved_file)
        sp.save_npz(saved_file, list_transition_matrix[i])

    mc_model = MarkovChain(item_dict, reversed_item_dict, item_freq_dict, w_behavior, list_transition_matrix, mc_order)

def create_lines_from_df(groupby_date_df):
    user_behavior_dict = dict()
    for i in range(len(groupby_date_df)):
        uid = groupby_date_df.loc[i, 'user_id']
        basket = groupby_date_df.loc[i, 'list ib']
        if uid not in user_behavior_dict:
            user_behavior_dict[uid] = []
            user_behavior_dict[uid].append(basket)
        else:
            user_behavior_dict[uid].append(basket)
    lines = []
    for u in user_behavior_dict:
        line = str(u)
        b_seq = user_behavior_dict[u]
        for basket in b_seq:
            line += ' |'
            for ib_pair in basket:
                line += (' ' + str(ib_pair[0]) + ':' + str(ib_pair[1]))
        lines.append(line)
    return lines


def filter_target_behavior(lines, mc_order, target_behavior, filter_string):
    filtered_lines = []
    for line in lines:
        elements = line.split("|")
        user = elements[0]
        basket_seq = elements[1:]
        if len(basket_seq) < 2:
            continue
        st = 1
        for i in range(st, len(basket_seq)):
            filtered_line = user
            cur_basket = basket_seq[i]
            if target_behavior not in cur_basket:
                continue
            if i + 1 < mc_order:
                prev_baskets = basket_seq[i - st:i]
            else:
                prev_baskets = basket_seq[i - mc_order:i]

            for i in range(0, len(prev_baskets)):
                prev_baskets[i] = re.sub(filter_string, '', prev_baskets[i])
                if len(prev_baskets[i].strip()) > 1:
                    filtered_line += ("|" + prev_baskets[i])

            #             filtered_line += "|".join(prev_baskets)
            filtered_line += "|"
            cur_item_list = [p for p in re.split('[\\s]+', cur_basket.strip())]
            for item in cur_item_list:
                if target_behavior in item:
                    filtered_line += (' ' + item)
            if len(filtered_line.split("|")) > 2:
                filtered_lines.append(filtered_line)
    return filtered_lines

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', help='The directory of input', type=str, default='../data/')
    parser.add_argument('--output_dir', help='The directory of output', type=str, default='../saved_models/')
    parser.add_argument('--target_behavior', help='Target behavior', type=str, default='buy')

    args = parser.parse_args()

    data_dir = args.input_dir
    o_dir = args.output_dir
    target_behavior = args.target_behavior
    csv_file = glob.glob(data_dir+'*.csv')
    # u_behavior_df = pd.read_csv(csv_file, nrows=1000, names=['user_id', 'item_id', 'category_id', 'behavior', 'time_stamp'])
    u_behavior_df = pd.read_csv(csv_file)
    new_behavior_df = u_behavior_df.sort_values(['user_id', 'date']).drop(['category_id'], axis=1)
    new_behavior_df['pair_ib'] = new_behavior_df[['item_id', 'behavior']].apply(tuple, axis=1)
    new_behavior_df.drop(['item_id', 'behavior'], axis=1, inplace=True)
    groupby_date_df = new_behavior_df.groupby(['user_id', 'date'])['pair_ib'].apply(list).reset_index(name='list ib')
    raw_lines = create_lines_from_df(groupby_date_df)
    mc_order = 1

    # model for recommend buy target
    if target_behavior == 'buy':
        filter_string = '[\\s][0-9]+:fav|[\\s][0-9]+:buy'
        filtered_lines = filter_target_behavior(raw_lines, mc_order, target_behavior, filter_string)
        model_name = 'buy_seqrec'
        train_model(filtered_lines, model_name, o_dir)
    # model for recommend cart target
    elif target_behavior == 'cart':
        filter_string = '[\\s][0-9]+:buy'
        filtered_lines = filter_target_behavior(raw_lines, mc_order, target_behavior, filter_string)
        model_name = 'cart_seqrec'
        train_model(filtered_lines, model_name, o_dir)
    # model for recommend pv target
    elif target_behavior == 'pv':
        filter_string = '[\\s][0-9]+:buy|[\\s][0-9]+:fav|[\\s][0-9]+:cart'
        filtered_lines = filter_target_behavior(raw_lines, mc_order, target_behavior, filter_string)
        model_name = 'pv_seqrec'
        train_model(filtered_lines, model_name, o_dir)
    else:
        print("Don't use this behavior")
        sys.exit(0)

