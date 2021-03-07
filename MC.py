import scipy.sparse as sp
import numpy as np

class MarkovChain():
  def __init__(self, item_dict, reversed_item_dict, item_freq_dict, weight_behaivor, list_transition_matrix, mc_order):
    self.item_freq_dict = item_freq_dict
    self.item_dict = item_dict
    self.reversed_item_dict = reversed_item_dict
    self.nb_items = len(item_dict)
    # self.sp_matrix_path = sp_matrix_path
    self.mc_order = mc_order
    self.w_behavior = weight_behaivor
    self.list_transition_matrix = list_transition_matrix

  def top_predicted_item(self, previous_basket, topk):
    candidate = np.zeros(self.nb_items)
    prev_basket_idx = [self.item_dict[item] for item in previous_basket]
    # for item_idx in prev_basket_idx:
    # for i in range(len(self.behavior_dict)):
    candidate = np.array(self.transition_matrix[prev_basket_idx, :].todense().sum(axis=0))[0]
    candidate = candidate / len(prev_basket_idx)
    topk_idx = np.argpartition(candidate, -topk)[-topk:]
    sorted_topk_idx = topk_idx[np.argsort(candidate[topk_idx])]
    topk_item = [self.reversed_item_dict[item] for item in sorted_topk_idx]
    # print("Done")
    return topk_item

  def top_predicted_mc_order(self, previous_baskets, topk):
    total_score = np.zeros(self.nb_items)
    nb_previous_basket = len(previous_baskets)
    # combine_idx = []
    # combine_score = []
    for i in range(nb_previous_basket - 1, -1, -1):
      prev_basket_idx = [self.item_dict[item] for item in previous_baskets[i]]
      candidate = np.array(self.list_transition_matrix[nb_previous_basket-i-1][prev_basket_idx, :].todense().sum(axis=0))[0]
      candidate = candidate / len(prev_basket_idx)
      total_score += candidate

    topk_idx = np.argpartition(total_score, -topk)[-topk:]
    sorted_topk_idx = topk_idx[np.argsort(total_score[topk_idx])]
    topk_item = [self.reversed_item_dict[item] for item in sorted_topk_idx]
    return topk_item