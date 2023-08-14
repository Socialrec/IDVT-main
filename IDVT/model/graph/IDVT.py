import torch
import torch.nn as nn
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE_inter,  InfoNCE_intra, InfoNCE_inter_intra
from data.social import Relation
from data.augmentor import GraphAugmentor
torch.set_printoptions(profile="full")

class IDVT(GraphRecommender):
    def __init__(self, conf, training_set, test_set, **kwargs):
        super(IDVT, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['IDVT'])
        self.n_layers = int(args['-n_layer'])
        self.cl_rate = float(args['-lambda1'])
        self.c2_rate = float(args['-lambda2'])
        aug_type = self.aug_type = int(args['-augtype'])
        drop_rate = float(args['-droprate'])
        temp1 = float(args['-temp1'])
        temp2 = float(args['-temp2'])
        self.social_data = Relation(conf, kwargs['social.data'], self.data.user)
        self.model = IDVT_Encoder(self.data, self.emb_size, self.n_layers, self.social_data, drop_rate, temp1, temp2, aug_type)

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            droped_social1 = model.graph_reconstruction()
            droped_social2 = model.graph_reconstruction()
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb  = model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
               
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                cl_loss = self.cl_rate * model.cal_cl_loss([user_idx,pos_idx], droped_social1 , droped_social2)
                c2_loss = self.c2_rate * model.cal_c2_loss([user_idx,pos_idx],rec_user_emb)
                batch_loss =  rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb,neg_item_emb) + cl_loss + c2_loss

                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 50==0 and n>0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss', cl_loss.item(), 'c2_loss', c2_loss.item())
                    
            with torch.no_grad():
                self.user_emb, self.item_emb  = model()
            if epoch % 5 == 0:
                self.fast_evaluation(epoch)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb


    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()

class IDVT_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers , social_data, drop_rate , temp1, temp2, aug_type):
        super(IDVT_Encoder, self).__init__()

        self.data = data
        self.social_data = social_data
        self.latent_size = emb_size
        self.layers = n_layers
        self.drop_rate = drop_rate
        self.temp1 = temp1
        self.temp2 = temp2
        self.aug_type = aug_type
        self.norm_adj = data.norm_adj
        self.social_adj = self.social_data.get_social_mat()

        #self.case_study(self.social_adj,self.data.interaction_mat,4870,4861)
        #self.case_study(self.social_adj,self.data.interaction_mat,4870,4791)

        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()
        self.socialGraph = TorchGraphInterface.convert_sparse_mat_to_tensor(self.social_adj).cuda()
        self.Graph_Comb_u = Graph_Comb_user_EAGCN(self.latent_size)
        self.mean_sim = 0
        self.pruning = 0

    def sp_cos_sim(self, a, b, social_adj ,eps=1e-8):
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n)) 
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))

        social_indice = social_adj.coalesce().indices()
        social_shape  = social_adj.coalesce().shape

        L = social_indice.shape[1]  # self.social_indice.shape: torch.Size([2, 119728])
        sims = torch.zeros(L, dtype=a.dtype).cuda()

        a_batch = torch.index_select(a_norm, 0, social_indice[0, :])
        b_batch = torch.index_select(b_norm, 0, social_indice[1, :])
        dot_prods = torch.mul(a_batch, b_batch).sum(1)
        sims[:] = dot_prods

        return torch.sparse.FloatTensor(social_indice, sims, size=social_shape).coalesce()

    def get_sim_adj(self , u_emb , social_adj):

        social_shape  = social_adj.coalesce().shape
        u_emb = torch.sparse.mm(social_adj, u_emb)
        sim_adj = self.sp_cos_sim(u_emb, u_emb , social_adj)

        # pruning
        sim_value = torch.div(torch.add(sim_adj.values(), 1), 2)  # sim = ( sim + 1 ) /2
        mean_sim = torch.mean(sim_value)
        #pruning = mean_sim
        pruning = 0
        if ( mean_sim > 0.7) :
                pruning = 0.8

        self.mean_sim = mean_sim
        self.pruning = pruning
        self.filter_num = (sim_value < pruning).sum().item()

        # torch.where(condition，a，b) 输入参数condition：条件限制，如果满足条件，则选择a，否则选择b作为输出
        pruned_sim_value = torch.where(sim_value < pruning, torch.zeros_like(sim_value),sim_value) if pruning > 0 else sim_value
        pruned_sim_adj = torch.sparse.FloatTensor(sim_adj.indices(), pruned_sim_value, social_shape).coalesce()
        self.pruned_sim_adj = pruned_sim_adj

        # normalize
        pruned_sim_indices = pruned_sim_adj.indices()
        diags = torch.sparse.sum(pruned_sim_adj, dim=1).to_dense() + 1e-7
        diags = torch.pow(diags, -1)
        diag_lookup = diags[pruned_sim_indices[0, :]]

        pruned_sim_adj_value = pruned_sim_adj.values()
        normal_sim_value = torch.mul(pruned_sim_adj_value, diag_lookup)
        normal_sim_adj = torch.sparse.FloatTensor(pruned_sim_indices, normal_sim_value, social_shape).cuda().coalesce()


        return normal_sim_adj

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        return embedding_dict

    def graph_reconstruction(self):
        if self.aug_type==0 or 1:
            dropped_adj = self.random_graph_augment()
        else:
            dropped_adj = []
            for k in range(self.n_layers):
                dropped_adj.append(self.random_graph_augment())
        return dropped_adj

    def random_graph_augment(self):
        dropped_mat = None
        if self.aug_type == 0:
            dropped_mat = GraphAugmentor.node_dropout(self.social_adj, self.drop_rate)
        elif self.aug_type == 1 or self.aug_type == 2:
            dropped_mat = GraphAugmentor.edge_dropout(self.social_adj, self.drop_rate)
        return TorchGraphInterface.convert_sparse_mat_to_tensor(dropped_mat).cuda()

    def forward(self, perturbed_adj=None , ui=False):

        if (perturbed_adj is not None) :

            S = self.get_sim_adj(self.embedding_dict['user_emb'] , perturbed_adj)

            user_emb_social_set = []
            users_emb_1 = self.embedding_dict['user_emb']
            for layer in range(3):
                user_emb_temp_1= torch.sparse.mm(S, users_emb_1)
                user_emb_social_set.append(user_emb_temp_1)
                users_emb_1 = user_emb_temp_1
            user_sview_emb = torch.stack(user_emb_social_set, dim=1)
            user_sview_emb = torch.mean(user_sview_emb, dim=1)

            ego_embeddings_v1 = torch.cat([self.embedding_dict['user_emb'] + user_sview_emb, self.embedding_dict['item_emb']], 0)
            all_embeddings_v1 = [ego_embeddings_v1]
            for k in range(self.layers):
                ego_embeddings_v1 = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings_v1)
                all_embeddings_v1.append(ego_embeddings_v1)

            all_embeddings_v1 = torch.stack(all_embeddings_v1, dim=1)
            all_embeddings_v1 = torch.mean(all_embeddings_v1, dim=1)
            user_all_embeddings_v1 = all_embeddings_v1[:self.data.user_num]
            item_all_embeddings_v1 = all_embeddings_v1[self.data.user_num:]

            user_all_embeddings = self.Graph_Comb_u(user_all_embeddings_v1 , user_sview_emb)
            item_all_embeddings = item_all_embeddings_v1

        elif (ui==False):
            
            S = self.get_sim_adj(self.embedding_dict['user_emb'] , self.socialGraph)

            user_emb_social_set = []
            users_emb_1 = self.embedding_dict['user_emb']
            for layer in range(3):
                user_emb_temp_1= torch.sparse.mm(S, users_emb_1)
                user_emb_social_set.append(user_emb_temp_1)
                users_emb_1 = user_emb_temp_1
            user_sview_emb = torch.stack(user_emb_social_set, dim=1)
            user_sview_emb = torch.mean(user_sview_emb, dim=1)


            ego_embeddings_v1 = torch.cat([self.embedding_dict['user_emb'] + user_sview_emb, self.embedding_dict['item_emb']], 0)
            all_embeddings_v1 = [ego_embeddings_v1]
            for k in range(self.layers):
                ego_embeddings_v1 = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings_v1)
                all_embeddings_v1.append(ego_embeddings_v1)

            all_embeddings_v1 = torch.stack(all_embeddings_v1, dim=1)
            all_embeddings_v1 = torch.mean(all_embeddings_v1, dim=1)
            user_all_embeddings_v1 = all_embeddings_v1[:self.data.user_num]
            item_all_embeddings_v1 = all_embeddings_v1[self.data.user_num:]


            user_all_embeddings = self.Graph_Comb_u(user_all_embeddings_v1 , user_sview_emb)
            item_all_embeddings = item_all_embeddings_v1

        else:

            ego_embeddings_v1 = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
            all_embeddings_v1 = [ego_embeddings_v1]
            for k in range(self.layers):
                ego_embeddings_v1 = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings_v1)
                all_embeddings_v1.append(ego_embeddings_v1)

            all_embeddings_v1 = torch.stack(all_embeddings_v1, dim=1)
            all_embeddings_v1 = torch.mean(all_embeddings_v1, dim=1)
            user_all_embeddings_v1 = all_embeddings_v1[:self.data.user_num]
            item_all_embeddings_v1 = all_embeddings_v1[self.data.user_num:]

            user_all_embeddings = user_all_embeddings_v1
            item_all_embeddings = item_all_embeddings_v1

        return user_all_embeddings, item_all_embeddings 

    def cal_cl_loss(self, idx, perturbed_mat1, perturbed_mat2):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        user_view_1, item_view_1 = self.forward(perturbed_mat1)
        user_view_2, item_view_2 = self.forward(perturbed_mat2)
        view1 = torch.cat((user_view_1[u_idx],item_view_1[i_idx]),0)
        view2 = torch.cat((user_view_2[u_idx],item_view_2[i_idx]),0)

        #only user
        user_cl_loss = InfoNCE_inter(user_view_1[u_idx], user_view_2[u_idx], self.temp1)  
        return user_cl_loss 

    def cal_c2_loss(self, idx, user_view_1):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()

        user_view_2 ,items_view2 = self.forward(ui=True)
        ratio = 0.7
        user_c2_loss = ratio * InfoNCE_inter(user_view_1[u_idx], user_view_2[u_idx], self.temp2) + (1-ratio) * InfoNCE_intra(user_view_1[u_idx], self.temp2)
        return user_c2_loss

class Graph_Comb_user_EAGCN(nn.Module):
    def __init__(self, embed_dim):
        super(Graph_Comb_user_EAGCN, self).__init__()
        self.gate1 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.gate2 = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x, y):
        gu = torch.sigmoid(self.gate1(x)+self.gate2(y))
        output = torch.mul(gu,y)+torch.mul((1-gu),x)
        return output



