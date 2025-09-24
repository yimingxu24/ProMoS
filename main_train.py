import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from utils import Dataset
from torch_geometric.utils import to_undirected
from torch_scatter import scatter_mean
import faiss
from typing import List
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import random
import csv
import math
import os
import time
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


class MLPExpert(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)


class Router(nn.Module):
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.num_experts = num_experts
        self.route = nn.Linear(input_dim, num_experts)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.route.weight, gain=0.5)
        nn.init.zeros_(self.route.bias)
                
    def forward(self, x):   
        logits = self.route(x)

        top_k_scores, top_k_indices = torch.topk(logits, 2, dim=1)

        probabilities = F.softmax(top_k_scores, dim=1)

        return top_k_indices.squeeze(-1), probabilities


def faiss_kmeans_init(x_tensor: torch.Tensor, cluster_nums: List[int]) -> nn.ParameterList:
    x_np = x_tensor.cpu().numpy().astype('float32')
    dim = x_np.shape[1]
    prototypes = []

    for k in cluster_nums:
        kmeans = faiss.Kmeans(d=dim, k=k, niter=20, verbose=False)
        kmeans.train(x_np)
        centroids = torch.from_numpy(kmeans.centroids).to(x_tensor.device)

        prototypes.append(nn.Parameter(centroids))
    return nn.ParameterList(prototypes)



def combine_faiss_and_xavier_init(t_feat: torch.Tensor, cluster_nums: List[int], dim: int) -> nn.ParameterList:
    faiss_prototypes = faiss_kmeans_init(t_feat, cluster_nums)
    combined = nn.ParameterList()

    for proto in faiss_prototypes:
        n, d = proto.shape
        p = torch.randn(n, d, device=proto.device)

        p = F.normalize(p, dim=1)

        proto.data = F.normalize(proto.data, dim=1) 

        avg_proto = proto.data * 0.5 + p * 0.5

        combined.append(nn.Parameter(avg_proto))

    return combined


class StudentMoE(nn.Module):
    def __init__(self, input_dim, t_feat=None, num_experts=20, expert_output_dim=64, 
                 prototypes_per_group_generalist=[20], prototypes_per_group_specialized=[20]):

        super().__init__()

        self.temperature = nn.Parameter(torch.tensor(2.0))
        self.num_experts = num_experts

        self.generalist_experts = nn.ModuleList([MLPExpert(input_dim, output_dim=expert_output_dim) for _ in range(1)])

        self.adapter = nn.Linear(expert_output_dim, expert_output_dim)

        self.specialized_experts = nn.ModuleList([MLPExpert(input_dim, output_dim=expert_output_dim) for _ in range(num_experts)])  
        self.router_specialized = Router(input_dim, num_experts=num_experts)

        self.prototype_groups_generalist = combine_faiss_and_xavier_init(
            t_feat, prototypes_per_group_generalist, expert_output_dim
        )
        self.prototype_groups_specialized = combine_faiss_and_xavier_init(
            t_feat, prototypes_per_group_specialized, expert_output_dim
        )


        self.prototypes_per_group_generalist = prototypes_per_group_generalist
        self.prototypes_per_group_specialized = prototypes_per_group_specialized
        self.dropout = nn.Dropout(0.1)


    def adp(self, x):
        return self.adapter(x)

    def forward(self, x):
        expert_outputs_generalist = torch.zeros(x.size(0), 64).to(x.device)

        for i in range(len(self.generalist_experts)):
            expert_outputs_generalist = self.generalist_experts[i](self.dropout(x))

        expert_indices, probabilities = self.router_specialized(x)

        expert_outputs_specialized = torch.zeros(x.size(0), 64).to(x.device)
      
        for i, expert in enumerate(self.specialized_experts):
            mask = (expert_indices == i)

            if mask.any():
                rows, cols = mask.nonzero(as_tuple=True)

                x_selected = x[rows]
                prob_selected = probabilities[rows, cols]
                out_selected = expert(self.dropout(x_selected)) * prob_selected.unsqueeze(1)

                expert_outputs_specialized.index_add_(0, rows, out_selected)

        expert_outputs_generalist = expert_outputs_generalist + x
        expert_outputs_specialized = expert_outputs_specialized + x

        distance_distributions_generalist = []
        for proto in self.prototype_groups_generalist:
            sim = _similarity(expert_outputs_generalist, proto)
            sim = F.softmax(sim / self.temperature, dim=-1)
            distance_distributions_generalist.append(sharpen(sim))
                    
        distance_distributions_specialized = []
        for proto in self.prototype_groups_specialized:
            sim = _similarity(expert_outputs_specialized, proto)
            sim = F.softmax(sim / self.temperature, dim=-1)
            distance_distributions_specialized.append(sharpen(sim))

            
        return distance_distributions_generalist, distance_distributions_specialized, expert_outputs_generalist, expert_outputs_specialized


def compute_loss(student_dists, teacher_dists):
    loss = 0
    for s_dist, t_dist in zip(student_dists, teacher_dists):
        log_s = torch.log(s_dist + 1e-8)

        loss += F.kl_div(log_s, t_dist, reduction='batchmean')

    return loss

def to_edge_index(adj):
    if isinstance(adj, torch.Tensor):
        coo = adj.coalesce()
        return coo.indices()
    else:
        raise ValueError("Unsupported adjacency format")

@torch.no_grad()
def compute_kl_score(student_dists, teacher_dists):
    score = 0
    for s, t in zip(student_dists, teacher_dists):
        log_s = torch.log(s + 1e-8)
        kl = F.kl_div(log_s, t, reduction='none').sum(dim=1)  # [N]
        score += kl        

    return score


def evaluate_anomaly(scores, labels):
    scores = scores.cpu().numpy()
    labels = labels.cpu().numpy()
    auroc = roc_auc_score(labels, scores)
    auprc = average_precision_score(labels, scores)
    return auroc, auprc

def _similarity(h1, h2):
    h1_norm = (h1 ** 2).sum(dim=1)
    h2_norm = (h2 ** 2).sum(dim=1)
    dist_sq = h1_norm[:, None] + h2_norm[None, :] - 2 * h1 @ h2.T

    return -dist_sq

def sharpen(p, T=1):
    # sharp_p = p ** (1. / T)
    # sharp_p /= torch.sum(sharp_p, dim=1, keepdim=True)
    # return sharp_p
    return p


def residual_features(x, edge_index):
    row, col = edge_index

    neigh_feat = scatter_mean(x[col], row, dim=0, dim_size=x.size(0))

    res_feat = x - neigh_feat
    return res_feat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher_feat_path', type=str, default='./emb/')
    parser.add_argument('--teacher_model', type=str, default='gca')
    parser.add_argument('--datasets', nargs='+', default=['pubmed', 'Flickr', 'questions', 'YelpChi'])

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--proto_lr', type=float, default=5e-3)

    parser.add_argument('--num_prototypes', type=int, default=20)
    parser.add_argument('--num_experts', type=int, default=20)

    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--mu', type=float, default=0.6)
    parser.add_argument('--lam', type=float, default=0.5)


    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--dim', type=int, default=64)

    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    datasets = []
    node_counts = []
    for name in args.datasets:
        ds = Dataset(dims=args.dim, name=name)
        data = ds.graph
        data.edge_index = to_undirected(to_edge_index(data.adj))
        datasets.append(data)
        node_counts.append(data.x.size(0))


    teacher_feats = []
    for name in args.datasets:
        teacher_feat_path = os.path.join(args.teacher_feat_path, f'{args.teacher_model}/{name}.pt')
        t_feats = torch.load(teacher_feat_path).to(device)
        teacher_feats.append(t_feats)


    teacher_feat_all = torch.cat(teacher_feats, dim=0).to(device)
    input_dim = teacher_feats[0].size(1)

    auc_dict = {}
    pre_dict = {}

    training_time, eval_time = 0, 0

    for t in range(args.trials):
        args.seed = t
        # Set random seed
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        random.seed(args.seed)

        model = StudentMoE(input_dim=args.dim, t_feat=teacher_feat_all, num_experts=args.num_experts, 
        prototypes_per_group_generalist=[args.num_prototypes], prototypes_per_group_specialized=[args.num_prototypes]).to(device)

        main_params, proto_params = [], []

        for name, param in model.named_parameters():
            if 'prototype' in name:
                proto_params.append(param)
            else:
                main_params.append(param)


        optimizer = torch.optim.Adam([
            {'params': main_params, 'lr': args.lr},
            {'params': proto_params, 'lr': args.proto_lr}
        ], weight_decay=5e-5)

        model.train()
        start_training = time.time()

        for epoch in range(args.epochs):
            total_loss, loss = 0, 0

            for i, data in enumerate(datasets):
                x = data.x.to(device)
                labels = data.ano_labels.long().to(device)
                edge_index = data.edge_index.to(device)

                t_feat = teacher_feats[i].detach()

                res_feat = residual_features(x, edge_index)

                t_feat = model.adp(t_feat)
                student_dists_gen, student_dists_spe, e_g, e_s = model(res_feat + x)

                teacher_dists_gen = []
                vq_loss, vq_loss1 = 0.0, 0.0
                
                for proto in model.prototype_groups_generalist:
                    sim = _similarity(t_feat, proto)
                    sim = F.softmax(sim / model.temperature, dim=-1)
                    teacher_dists_gen.append(sharpen(sim))

                    max_sim, nearest_idx = sim.max(dim=1)
                    nearest_proto = proto[nearest_idx]

                    pro_sim = F.softmax(_similarity(proto, proto) / model.temperature, dim=-1)

                    target_struct = pro_sim[nearest_idx]
                    kl_res = F.kl_div(sim.log(), target_struct, reduction='none').sum(dim=1).detach()

                    raw_weight = torch.sigmoid(-args.beta * (kl_res - args.mu))

                    weights = raw_weight / (raw_weight.sum() + 1e-8)

                    mse = F.mse_loss(nearest_proto, t_feat.detach(), reduction='none').sum(dim=1)
                    mse1 = F.mse_loss(nearest_proto.detach(), t_feat, reduction='none').sum(dim=1)
                    vq_loss += (weights * mse).sum() + (weights * mse1).sum()


                teacher_dists_spe = []
                for proto in model.prototype_groups_specialized:
                    sim = _similarity(t_feat, proto)
                    sim = F.softmax(sim / model.temperature, dim=-1)
                    teacher_dists_spe.append(sharpen(sim))

                    max_sim, nearest_idx = sim.max(dim=1)
                    nearest_proto = proto[nearest_idx]

                    pro_sim = F.softmax(_similarity(proto, proto) / model.temperature, dim=-1)

                    target_struct = pro_sim[nearest_idx]
                    kl_res = F.kl_div(sim.log(), target_struct, reduction='none').sum(dim=1).detach()

                    raw_weight = torch.sigmoid(-args.beta * (kl_res - args.mu))

                    weights = raw_weight / (raw_weight.sum() + 1e-8)

                    mse = F.mse_loss(nearest_proto, t_feat.detach(), reduction='none').sum(dim=1)
                    mse1 = F.mse_loss(nearest_proto.detach(), t_feat, reduction='none').sum(dim=1)
                    vq_loss1 += (weights * mse).sum() + (weights * mse1).sum()

                loss =  compute_loss(student_dists_gen, teacher_dists_gen) + compute_loss(student_dists_spe, teacher_dists_spe) + (vq_loss + vq_loss1) * args.lam # 4

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch}, Avg Loss: {total_loss / len(datasets):.4f}")
            end_training = time.time()

            training_time += end_training - start_training

        datasets_test = ['cora', 'citeseer', 'ACM', 'BlogCatalog', 'Facebook', 'weibo', 'Reddit', 'cs', 'photo', 'tolokers', 'tfinance']

        model.eval()

        all_scores, all_labels = {}, {}

        start_eval = time.time()

        for i, name in enumerate(datasets_test):
            ds = Dataset(dims=64, name=name)
            data = ds.graph
            data.edge_index = to_undirected(to_edge_index(data.adj))
            x = data.x.to(device)
            y = data.ano_labels.to(device)
            n = x.shape[0]

            data.edge_index = data.edge_index.to(device)
            edge_index = data.edge_index.to(device)

            anomaly_scores = 0

            teacher_feat_path = os.path.join(args.teacher_feat_path, f'{args.teacher_model}/{name}.pt')
            teacher_feat = torch.load(teacher_feat_path).to(device)

            res_feat = residual_features(x, edge_index)

            teacher_feat = model.adp(teacher_feat)

            student_dists_gen, student_dists_spe, e_g, e_s = model(res_feat + x)

            teacher_dists_gen = []
            vq_loss = 0
            for idx, proto in enumerate(model.prototype_groups_generalist):
                sim = _similarity(teacher_feat, proto)
                sim = F.softmax(sim / model.temperature, dim=-1)
                teacher_dists_gen.append(sharpen(sim))

                max_sim, nearest_idx = sim.max(dim=1)
                nearest_proto = proto[nearest_idx]

                mse = F.mse_loss(nearest_proto, teacher_feat.detach(), reduction='none')
                mse = mse.mean(dim=1).detach()

                max_sim, nearest_idx = student_dists_gen[idx].max(dim=1)
                nearest_proto = proto[nearest_idx]
                mse1 = F.mse_loss(nearest_proto, e_g.detach(), reduction='none')
                mse1 = mse1.mean(dim=1).detach()

                vq_loss += (mse + mse1) / 2


            teacher_dists_spe = []
            vq_loss1 = 0
            for idx, proto in enumerate(model.prototype_groups_specialized):
                sim = _similarity(teacher_feat, proto)
                sim = F.softmax(sim / model.temperature, dim=-1)
                teacher_dists_spe.append(sharpen(sim))

                max_sim, nearest_idx = sim.max(dim=1)
                nearest_proto = proto[nearest_idx]

                mse = F.mse_loss(nearest_proto, teacher_feat.detach(), reduction='none')
                mse = mse.mean(dim=1).detach()

                max_sim, nearest_idx = student_dists_spe[idx].max(dim=1)
                nearest_proto = proto[nearest_idx]
                mse1 = F.mse_loss(nearest_proto, e_s.detach(), reduction='none')
                mse1 = mse1.mean(dim=1).detach()

                vq_loss1 += (mse + mse1) / 2
            

            anomaly_scores = compute_kl_score(student_dists_gen, teacher_dists_gen) + compute_kl_score(student_dists_spe, teacher_dists_spe) + (vq_loss + vq_loss1) * args.lam
            auroc, auprc = evaluate_anomaly(anomaly_scores, y)

        
            if torch.is_tensor(anomaly_scores):
                anomaly_scores = anomaly_scores.cpu().numpy()
            if torch.is_tensor(y):
                y = y.detach().cpu().numpy()

            all_scores[name] = anomaly_scores
            all_labels[name] = y

            if name not in auc_dict:
                auc_dict[name] = []
                pre_dict[name] = []
            auc_dict[name].append(auroc)
            pre_dict[name].append(auprc)

            print(f"[{name}] AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}")

        end_eval = time.time()
        eval_time = end_eval - start_eval



    auc_mean_dict, auc_std_dict, pre_mean_dict, pre_std_dict = {}, {}, {}, {}

    for test_data_name in auc_dict:
        auc_mean_dict[test_data_name] = np.mean(auc_dict[test_data_name])
        auc_std_dict[test_data_name] = np.std(auc_dict[test_data_name])
        pre_mean_dict[test_data_name] = np.mean(pre_dict[test_data_name])
        pre_std_dict[test_data_name] = np.std(pre_dict[test_data_name])
    for test_data_name in auc_mean_dict:
        str_result = 'AUROC:{:.4f}+-{:.4f}, AUPRC:{:.4f}+-{:.4f}'.format(
            auc_mean_dict[test_data_name],
            auc_std_dict[test_data_name],
            pre_mean_dict[test_data_name],
            pre_std_dict[test_data_name])
        print('-' * 50 + test_data_name + '-' * 50)
        print('str_result', str_result)

    average_auroc = np.mean([auc_mean_dict[name] for name in auc_mean_dict])
    print('\nAverage AUROC over selected datasets: {:.4f}'.format(average_auroc))


    average_auprc = np.mean([pre_mean_dict[name] for name in pre_mean_dict])
    print('\nAverage AUPRC over selected datasets: {:.4f}'.format(average_auprc))

if __name__ == '__main__':
    main()
