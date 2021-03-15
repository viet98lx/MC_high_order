import re
import numpy as np
import MC_utils
from MC import MarkovChain
import argparse
import scipy.sparse as sp
import os
import json

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', help='The directory of input', type=str, default='../data/')
    parser.add_argument('--output_dir', help='The directory of output', type=str, default='../saved_models/')
    parser.add_argument('--model_name', help='Model name ', type=str, default='mc')
    parser.add_argument('--mc_order', help='Markov order', type=int, default=1)
    parser.add_argument('--w_behavior', help='Weight behavior file', type=str, default=None)
    parser.add_argument('--toy_split', help='Ratio split', type=float, default=1)
    # parser.add_argument('--predict_file', help='predict result file', type=str, default=None)
    args = parser.parse_args()

    data_dir = args.input_dir
    o_dir = args.output_dir
    model_name = args.model_name
    mc_order = args.mc_order
    w_behavior_file = args.w_behavior
    toy_ratio = args.toy_split
    # predict_file = args.predict_file

    train_data_path = data_dir+'train_lines.txt'
    train_instances = MC_utils.read_instances_lines_from_file(train_data_path)
    nb_train = len(train_instances)
    print(nb_train)

    test_data_path = data_dir+'test_lines.txt'
    test_instances = MC_utils.read_instances_lines_from_file(test_data_path)
    nb_test = len(test_instances)
    print(nb_test)

    split_train = int(toy_ratio*nb_train)
    # split_test = int(0.5*nb_test)

    train_instances = train_instances[:split_train]
    # test_instances = test_instances[:split_test]

    if w_behavior_file is None:
        w_behavior = {'buy': 1, 'cart': 1, 'fav': 1, 'pv': 1}
    else:
        with open(w_behavior_file, 'r') as fp:
            w_behavior = json.load(fp)

    ### build knowledge ###
    # common_instances = train_instances + test_instances
    print("---------------------@Build knowledge-------------------------------")
    MAX_SEQ_LENGTH, item_dict, reversed_item_dict, item_probs, item_freq_dict, user_dict = MC_utils.build_knowledge(train_instances+test_instances, w_behavior)
    print('Build knowledge done')
    list_transition_matrix = MC_utils.calculate_transition_matrix(train_instances, item_dict, item_freq_dict, reversed_item_dict, w_behavior, mc_order)

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
        sp_matrix_path = model_name+'_transition_matrix_MC_'+str(i+1)+ '.npz'
        # nb_item = len(item_dict)
        # print('Density : %.6f' % (transition_matrix.nnz * 1.0 / nb_item / nb_item))
        saved_file = os.path.join(o_dir, sp_matrix_path)
        print("Save model in ", saved_file)
        sp.save_npz(saved_file, list_transition_matrix[i])

    mc_model = MarkovChain(item_dict, reversed_item_dict, item_freq_dict, w_behavior, list_transition_matrix, mc_order)
    topk = 50
    print('Predict to outfile')
    predict_file = os.path.join(o_dir, 'predict_'+model_name+'.txt')
    MC_utils.write_predict(predict_file, test_instances, topk, mc_model)
    print('Predict done')
    ground_truth, predict = MC_utils.read_predict(predict_file)
    for topk in [5, 10, 15, 20]:
        print("Top : ", topk)
        # hit_rate = MC_hit_ratio(test_instances, topk, mc_model)
        # recall = MC_recall(test_instances, topk, mc_model)
        hit_rate = MC_utils.hit_ratio(ground_truth, predict, topk)
        recall = MC_utils.recall(ground_truth, predict, topk)
        print("hit ratio: ", hit_rate)
        print("recall: ", recall)
