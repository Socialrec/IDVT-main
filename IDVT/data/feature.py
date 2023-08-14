from collections import defaultdict
from data.graph import Graph
import numpy as np
import scipy.sparse as sp

class Feature(Graph):
    def __init__(self, conf, feature, item):
        super().__init__()
        self.config = conf
        self.feature = feature
        self.item = item
        self.cat = {}
        self.id2cat = {}
        self.__generate_dict()
    
    def __generate_dict(self):
        idx = []
        for n, pair in enumerate(self.feature):
            if pair[0] not in self.item:
                idx.append(n)
        for item in reversed(idx):
            del self.feature[item]
            
        for entry in self.feature:
            item, cat, weight = entry
            if cat not in self.cat:
                self.cat[cat] = len(self.cat)
                self.id2cat[self.cat[cat]] = cat


    def get_item_cat_mat(self):
        row, col, entries = [], [], []
        for pair in self.feature:
            row += [self.item[pair[0]]]
            col += [self.cat[pair[1]]]
            entries += [1.0]
        item_cat_mat = sp.csr_matrix((entries, (row, col)), shape=(len(self.item), len(self.cat)), dtype=np.float32)
        return item_cat_mat 

    def get_cat_norm_adj(self,ui_adj):
        ic_adj = self.get_item_cat_mat()
        i_u_i_adj = ui_adj.T.dot(ui_adj)
        i_c_i_adj = ic_adj.dot(ic_adj.T)
        #print(i_u_i_adj.shape)
        #print(i_c_i_adj.shape)
        i_i_adj = i_u_i_adj.multiply(i_c_i_adj)
        nonzero_mask = np.array(i_i_adj[i_i_adj.nonzero()] > 1)[0]
        rows = i_i_adj.nonzero()[0][nonzero_mask]
        cols = i_i_adj.nonzero()[1][nonzero_mask]
        i_i_adj[rows, cols] = 1 
        i_i_norm_adj = self.normalize_graph_mat(i_i_adj)
        return i_i_norm_adj

    def get_cat_adj(self,ui_adj):
        ic_adj = self.get_item_cat_mat()
        i_u_i_adj = ui_adj.T.dot(ui_adj)
        i_c_i_adj = ic_adj.dot(ic_adj.T)
        #print(i_u_i_adj.shape)
        #print(i_c_i_adj.shape)
        i_i_adj = i_u_i_adj.multiply(i_c_i_adj)
        nonzero_mask = np.array(i_i_adj[i_i_adj.nonzero()] > 1)[0]
        rows = i_i_adj.nonzero()[0][nonzero_mask]
        cols = i_i_adj.nonzero()[1][nonzero_mask]
        i_i_adj[rows, cols] = 1 
        return i_i_adj






