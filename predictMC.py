import sys, os, pickle, argparse
import re
import random
import scipy.sparse as sp
import numpy as np
import MC_utils
from MC import MarkovChain
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', help='The directory of input', type=str, default='../data/')
    parser.add_argument('--output_dir', help='The directory of output', type=str, default='../saved_models/')
    parser.add_argument('--model_name', help='Model name ', type=str, default='fpmc')
    parser.add_argument('--nb_predict', help='# of predict', type=int, default=10)
    parser.add_argument('--topk', help='# of predict', type=int, default=10)
    parser.add_argument('--mc_order', help='Markov order', type=int, default=1)
    parser.add_argument('--example_file', help='Example_file', type=str, default=None)
    parser.add_argument('--w_behavior', help='Weight behavior file', type=str, default=None)
    args = parser.parse_args()

    f_dir = args.input_dir
    o_dir = args.output_dir
    model_name = args.model_name
    nb_predict = args.nb_predict
    topk = args.topk
    ex_file = args.example_file
    w_behavior_file = args.w_behavior
    mc_order = args.mc_order

    data_dir = f_dir
    train_data_path = data_dir + 'train_lines.txt'
    train_instances = MC_utils.read_instances_lines_from_file(train_data_path)
    nb_train = len(train_instances)
    # print(nb_train)

    test_data_path = data_dir + 'test_lines.txt'
    test_instances = MC_utils.read_instances_lines_from_file(test_data_path)
    nb_test = len(test_instances)

    if w_behavior_file is None:
        w_behavior = {'buy': 1, 'cart': 0.5, 'fav': 0.5, 'pv': 0.5}
    else:
        with open(w_behavior_file, 'r') as fp:
            w_behavior = json.load(fp)
    # print(nb_test)
    print("---------------------@Build knowledge-------------------------------")
    MAX_SEQ_LENGTH, item_dict, reversed_item_dict, item_probs, item_freq_dict, user_dict = MC_utils.build_knowledge(train_instances+test_instances,w_behavior)

    if not os.path.exists(o_dir):
        os.makedirs(o_dir)
    saved_file = os.path.join(o_dir, 'transition_matrix_MC.npz')
    # print("Save model in ", saved_file)
    transition_matrix = sp.load_npz(saved_file)
    mc_model = MarkovChain(item_dict, reversed_item_dict, item_freq_dict, w_behavior, transition_matrix, mc_order)

    if ex_file is not None:
        ex_instances = MC_utils.read_instances_lines_from_file(ex_file)
    else :
        ex_instances = test_instances
    for i in random.sample(ex_instances, nb_predict):
        elements = i.split('|')
        b_seq = elements[-mc_model.mc_order-1:-1]
        # prev_basket = [item for item in re.split('[\\s]+',b_seq[-2].strip())]
        prev_item = []
        for prev_basket in b_seq[:-1]:
            prev_item +=[p.split(':')[0] for p in re.split('[\\s]+', prev_basket.strip())]
        target_basket = [p.split(':')[0] for p in re.split('[\\s]+',b_seq[-1].strip())]
        topk_item = mc_model.top_predicted_item(prev_item, topk)
        correct_set = set(topk_item).intersection(set(target_basket))
        print("Input basket: ", prev_item)
        print("Ground truth: ", target_basket)
        print("Nb_correct: ", len(correct_set))
        print("Predict topk: ", topk_item)
        print("Items correct: ", list(correct_set))
        print()