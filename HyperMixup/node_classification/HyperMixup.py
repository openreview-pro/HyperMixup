
import time
from copy import deepcopy
import random

import itertools
from typing import List,Union

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
# from triton.language import tensor

from structure import Graph, Hypergraph
from dhg.data import Cora, Pubmed, Citeseer
from dhg.models import HGNN, HGNNP, HNHN
from dhg.random import set_seed
from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator

import wandb
use_cuda = torch.cuda.is_available()
# # 遍历稠密图，找到超边节点关系
def create_e_list_from_incidence_matrix(incidence_matrix: np.ndarray) -> List[Union[List[int], int]]:
    e_list = []
    num_hyperedges = incidence_matrix.shape[1]

    for hyperedge_index in range(num_hyperedges):
        # 找到属于当前超边的所有节点的索引
        nodes_in_hyperedge = np.where(incidence_matrix[:, hyperedge_index] == 1)[0]

        # 如果只有一个节点，直接添加节点索引；如果有多个节点，添加节点索引列表
        # if len(nodes_in_hyperedge) == 1:
        #     e_list.append(nodes_in_hyperedge[0])
        # else:
        #     e_list.append(nodes_in_hyperedge.tolist())
        e_list.append(nodes_in_hyperedge.tolist())

    return e_list

def generate_node_pairs(train_mask: torch.Tensor):
    # 获取训练集节点的索引
    train_indices = torch.nonzero(train_mask).squeeze()  # 获取所有True的位置索引

    # 使用 itertools 生成所有的两两节点对
    node_pairs = list(itertools.combinations(train_indices.tolist(), 2))

    return train_indices,node_pairs

# 给超图hg加节点，给X特征矩阵加节点，给标签张量加节点
def modify_incidence_matrix(hg: Hypergraph, x: torch.Tensor, lbl: torch.Tensor, p, q, l):

    # # 获取需要修改的超边索引和包含它们的节点对及其超边索引--同质性
    # # modify_hyper_index, max_pair_with_e_idx = hg.top20_percent_hyperedges_index()
    # # modify_hyper_index.sort()
    #
    #
    # # 1. 保留 max_pair_with_e_idx 中包含 modify_hyper_index 超边索引的节点对，删除其他节点对
    # filtered_pairs = []
    # # for pair, e_idx_list in max_pair_with_e_idx:
    # #     # 保留包含 modify_hyper_index 超边索引的节点对
    # #
    # #     if e_idx_list[0] in modify_hyper_index:
    # #         filtered_pairs.append((pair, e_idx_list))
    # for (u, v), e_idx_list in max_pair_with_e_idx:
    #     # 这里去除 e_idx_list 部分，直接保留节点对 (u, v)
    #     if any(e_idx in modify_hyper_index for e_idx in e_idx_list):
    #         filtered_pairs.append((u, v))  # 保留符合条件的节点对


    # 将超图的关联矩阵转换为稠密矩阵（dense）
    H_den = hg.H.to_dense()
    # ①训练集节点和节点的超边特征均值-只使用训练集两两节点对不可行
    train_indices, node_pairs = generate_node_pairs(train_mask)
    edge_features_mean = hg.calculate_Xe_mean_of_edge(train_indices, X)

    #②训练集两两节点对和同质性高的节点对求交集节点对
    # filtered_pairs_set = set(filtered_pairs)  # 转换为集合
    # node_pairs_set = set(node_pairs)  # 转换为集合
    # common_pairs = filtered_pairs_set.intersection(node_pairs_set)  # 获取交集
    #num_common_pairs = len(common_pairs)  # 共有节点对的数量

    #③使用训练集两两节点对相似度前n%的节点对
    similar_node_pairs = hg.calculate_top20_uv_similarity(node_pairs, X, l)

    # 新节点的超边关系将被添加到 H_den 的最后
    new_rows = []  # 存储每个新节点的超边关系

    # 动态扩展的新节点特征和标签列表
    new_node_features_list = []  # 存储新节点特征
    new_node_labels_list_a = []    # 存储新节点标签
    new_node_labels_list_b = []  # 存储新节点标签


    # 4. 为每个节点对生成新节点和其对应的超边关系
    for u, v in similar_node_pairs:
        # 计算 u 和 v 的共享超边
        shared_edges = hg.index_edges_between(u, v)  # 假设有这个方法来获取共享的超边
        new_row = torch.zeros(H_den.size(1), dtype=torch.float32)  # 每个新节点的超边关系存储在 new_row 中

        for e_idx in shared_edges:
            new_row[e_idx] = 1.0  # 新节点与共享超边的关系设置为1

        # 其次，对于父节点各自非共同连接的超边，使用常数 p 来选择连接多少超边
        u_edges = hg.N_e(u)
        v_edges = hg.N_e(v)

        # 计算 u 和 v 的非共享超边
        u_non_shared = list(set(u_edges) - set(shared_edges))
        v_non_shared = list(set(v_edges) - set(shared_edges))

        # 随机选择 p 的比例来连接这些非共享超边
        u_selected_edges = random.sample(u_non_shared, k=int(len(u_non_shared) * p)) if u_non_shared else []
        v_selected_edges = random.sample(v_non_shared, k=int(len(v_non_shared) * (1 - p))) if v_non_shared else []

        # 将选择的超边与新节点建立关系
        for e_idx in u_selected_edges + v_selected_edges:
            new_row[e_idx] = 1.0  # 新节点与这些超边的关系设置为1

        # 将该新节点的超边关系添加到 new_rows 中
        new_rows.append(new_row)

        # 5. 计算新节点的特征
        u_features = x[u, :]  # u 节点的特征
        v_features = x[v, :]  # v 节点的特征

        # 获取 u 和 v 对应的超边特征均值
        u_edge_mean = edge_features_mean.get(u, torch.zeros_like(x[0]))  # 默认为零向量
        v_edge_mean = edge_features_mean.get(v, torch.zeros_like(x[0]))  # 默认为零向量

        # 对于每个特征，按 p 和 (1-p) 计算加权和
        new_node_features = p * (q * u_features + (1 - q) * u_edge_mean) + (1 - p) * (q * v_features + (1 - q) * v_edge_mean)
        new_node_features_list.append(new_node_features)





        # 6. 计算新节点的标签

        #①将标签转化为概率限量
        # new_lbl = p * lbl[u] + (1 - p) * lbl[v]
        # new_node_labels_list.append(new_lbl)

        #②以p的大小界定新结点的标签
        # if p > 0.5:
        #     new_node_labels_list.append(lbl[u])  # 选择 u 节点的标签
        # else:
        #     new_node_labels_list.append(lbl[v])  # 选择 v 节点的标签

        #③从损失函数更改分为y_a,y_b
        new_node_labels_list_a.append(lbl[u])
        new_node_labels_list_b.append(lbl[v])


    # 7. 将所有的新节点的超边关系（new_rows）添加到 H_den 的最后
    H_den = torch.cat([H_den, torch.stack(new_rows)], dim=0)  # 新节点的超边关系被添加为多行

    # 8. 将新节点的特征添加到 x 的最后一行
    new_node_features_tensor = torch.stack(new_node_features_list)  # 转换为张量
    x = torch.cat([x, new_node_features_tensor], dim=0)  # 将新节点的特征添加到 x 的最后一行

    # 9. 将新节点的标签添加到 lbl 的最后一行
    new_node_labels_tensor = torch.tensor(new_node_labels_list_a, dtype=torch.long)  # 转换为张量
    lbl_a = torch.cat([lbl, new_node_labels_tensor], dim=0)  # 将新节点的标签添加到 lbl 的最后一行
    new_node_labels_tensor = torch.tensor(new_node_labels_list_b, dtype=torch.long)  # 转换为张量
    lbl_b = torch.cat([lbl, new_node_labels_tensor], dim=0)  # 将新节点的标签添加到 lbl 的最后一行

    # 10. 计算添加节点的数量
    num_insert_indices = len(similar_node_pairs)

    # 返回修改后的关联矩阵 H_den、特征矩阵 x 和标签 lbl 以及添加的节点数量
    return H_den, x, lbl_a, lbl_b,  num_insert_indices

def mixup_criterion(pred, y_a, y_b, lam):
    return lam * F.cross_entropy(pred, y_a) + (1 - lam) * F.cross_entropy(pred, y_b)

def train(net, X, G, lbls_a, lbls_b,lam_p, train_idx, optimizer, epoch):

    net.train()  # 设置模型为训练模式

    st = time.time()  # 记录开始时间
    optimizer.zero_grad()  # 清除梯度
    outs = net(X, G)  # 前向传播
    outs, lbls_a, lbls_b = outs[train_idx], lbls_a[train_idx], lbls_b[train_idx]  # 获取训练集的输出和标签
    #lbls = F.one_hot(lbls, num_classes=7).to(torch.float)

    #log_input = F.log_softmax(outs, dim=1)  # 对预测进行 log softmax
    #loss = F.kl_div(log_input, lbls, reduction='batchmean')  # 使用 KL 散度来计算损失

    #loss = torch.nn.CrossEntropyLoss(outs, lbls)  # 计算交叉熵损失
    # criterion = F.cross_entropy()  # 交叉熵损失函数
    loss = mixup_criterion(outs, lbls_a, lbls_b, lam_p)
    # loss = F.cross_entropy(outs, lbls)  # 计算交叉熵损失
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数
    print(f"Epoch: {epoch}, Time: {time.time() - st:.5f}s, Loss: {loss.item():.5f}")  # 打印当前轮次的信息

    return loss.item()  # 返回损失值

@torch.no_grad()
def infer(net, X, G, lbls, idx, test=False):

    net.eval()  # 设置模型为评估模式
    outs = net(X, G)  # 前向传播
    outs, lbls = outs[idx], lbls[idx]  # 获取指定索引的输出和标签
    if not test:
        res = evaluator.validate(lbls, outs)  # 验证模式
    else:
        res = evaluator.test(lbls, outs)  # 测试模式
    return res  # 返回评估结果

if __name__ == "__main__":

    print("\ntrain finished!11111")  # 打印训练完成信息
    set_seed(2022)  # 设置随机种子以保证结果可复现
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # 选择设备（GPU或CPU）
    evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])  # 创建评估器
    data = Cora()  # 加载Cora数据集
    # data = Pubmed()  # 加载Pubmed数据集（可选）
    # data = Citeseer()  # 加载Citeseer数据集（可选）
    X, lbl = data["features"], data["labels"]  # 获取特征和标签

    G = Graph(data["num_vertices"], data["edge_list"])  # 构建图
    HG = Hypergraph.from_graph_kHop(G, k=1)  # 从图构建超图
    # HG.add_hyperedges_from_graph_kHop(G, k=1)  # 添加1跳超边（可选）
    # HG = Hypergraph.from_graph_kHop(G, k=1)  # 从图构建超图（可选）
    # HG.add_hyperedges_from_graph_kHop(G, k=2, only_kHop=True)  # 添加2跳超边（可选）
    print("\ntrain finished!333333")  # 打印训练完成信息
    train_mask = data["train_mask"]  # 获取训练集掩码
    val_mask = data["val_mask"]  # 获取验证集掩码
    test_mask = data["test_mask"]  # 获取测试集掩码

    # low_label rate 低标签率实验
    # 设置比例 rate，例如 0.01
    # rate = 0.005
    # num_total_nodes = train_mask.size(0)
    # num_train_nodes = train_mask.sum().item()  # 140
    # num_select = int(num_total_nodes * rate)
    #
    # # 从 train_mask 为 True 的索引中随机选取 num_select 个
    # true_indices = train_mask.nonzero(as_tuple=True)[0].tolist()
    # selected_indices = random.sample(true_indices, num_select)
    #
    #
    # # 创建新的训练掩码
    # train_mask = torch.zeros_like(train_mask)
    # # selected_indices = [54, 107, 110, 26, 67, 61, 87, 95, 100, 37,
    # #                 89, 85, 21, 16, 63, 120, 43, 72, 28, 91,
    # #                 31, 135, 83, 105, 112, 126, 127]
    # train_mask[selected_indices] = True
    # print(f"原训练节点数: {num_train_nodes}")
    # print(f"选中的训练节点数: {train_mask.sum().item()}")
    #


    print("\n初始H的shape")  # 打印训练完成信息

    print(HG.H.to_dense().shape)
    # 生成新的HG
    #使用wandb
    # lam_p = wandb.config.lam_p
    # lam_q = wandb.config.lam_q
    # lam_l = wandb.config.lam_l
    # 不使用wandb
    lam_p = 0.49
    lam_q = 0.72
    lam_l = 0.0017

    H_pad, X, lbl_a, lbl_b, num_insert_indices = modify_incidence_matrix(HG, X,lbl, lam_p, lam_q, lam_l)
    print("\n生成新节点后H的shape")  # 打印训练完成信息
    print(H_pad.shape)
    print(X.shape)
    print(lbl_a.shape)
    print("\n添加了%d个节点" % num_insert_indices)  # 打印训练完成信息
    HG.num_v = HG.num_v + num_insert_indices
    e_list = create_e_list_from_incidence_matrix(H_pad.numpy())
    HG = Hypergraph(HG.num_v, e_list, device=device)

    # HG.v2e_aggregation(X) 超边的特征

    new_true_elements = torch.ones(num_insert_indices, dtype=torch.bool)
    new_false_elements = torch.zeros(num_insert_indices, dtype=torch.bool)

    # 使用 torch.cat 将最后一个元素添加到张量尾部
    train_mask = torch.cat((train_mask, new_true_elements), dim=0)
    val_mask = torch.cat((val_mask, new_false_elements), dim=0)
    test_mask = torch.cat((test_mask,new_false_elements), dim=0)







    net = HGNN(data["dim_features"], 16, data["num_classes"])  # 创建HGNN模型
    # net = HNHN(data["dim_features"], 16, data["num_classes"], use_bn=True)  # 创建HNHN模型（可选）
    # optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)  # 创建Adam优化器



    # Adam优化器同时优化模型的参数和p, q
    optimizer = optim.Adam(
        list(net.parameters()) ,  # 将模型参数和 p, q 一起加入优化器
        lr=0.01,
        weight_decay=5e-4
    )

    X, lbl_a, lbl_b = X.to(device), lbl_a.to(device), lbl_b.to(device)  # 将数据移动到设备
    HG = HG.to(X.device)  # 将超图移动到设备
    net = net.to(device)  # 将模型移动到设备

    best_state = None  # 初始化最佳状态
    best_epoch, best_val = 0, 0  # 初始化最佳轮次和最佳验证结果
    for epoch in range(200):  # 训练200轮
        # 训练
        train(net, X, HG, lbl_a, lbl_b, lam_p, train_mask, optimizer, epoch)
        # 验证
        if epoch % 1 == 0:
            with torch.no_grad():
                val_res = infer(net, X, HG, lbl_a, val_mask)  # 进行验证
            if val_res > best_val:
                print(f"update best: {val_res:.5f}")  # 打印最佳验证结果
                best_epoch = epoch  # 更新最佳轮次
                best_val = val_res  # 更新最佳验证结果
                best_state = deepcopy(net.state_dict())  # 保存最佳模型状态
    print("\ntrain finished!")  # 打印训练完成信息
    print(f"best val: {best_val:.5f}")  # 打印最佳验证结果
    # 测试
    print("test...")  # 打印测试信息
    net.load_state_dict(best_state)  # 加载最佳模型状态
    res = infer(net, X, HG, lbl_a, test_mask, test=True)  # 进行测试
    print(f"final result: epoch: {best_epoch}")  # 打印最终结果
    print(res)  # 打印测试结果


