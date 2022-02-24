import torch
import pickle as pkl
import json
import random
import numpy as np
import torch.nn.functional as F
import time

dbid2code = {}
# sys.setrecursionlimit(10000)
with open("dbid2code.json", "r") as f:
    dbid2code = json.load(f)

entityid2code = {}
with open("entityid2code.json", "r") as f:
    entityid2code = json.load(f)

cat_data = np.load("embedding_all.npz")
drug_embedding = cat_data['drug_embed']
entity_embedding = cat_data['entity_embed']
relation_embedding = cat_data['relation_embed']

graph = {}
with open("graph.pkl", "rb") as f:
    graph = pkl.load(f)

graph_relation = {}
with open("graph_relation.json", "r") as f:
    graph_relation = json.load(f)


def get_nerbs(entity, deepth, graph):
    results = [entity]
    N1 = graph[entity]
    while deepth > 0:
        N2 = []
        for each in N1:
            if each not in results:
                results.append(each)
                N2 += graph[each]
        N1 = N2
        deepth -= 1
    return results


def get_subgraph(nodes, graph):
    nodes = set(nodes)
    subgraph = {}
    for node in nodes:
        nerbs = set(graph[node])
        subgraph[node] = nodes & nerbs
    return subgraph


def findAllPath(graph, start, end, deepth):
    # 用一个栈记录中间节点，
    # 用一个list记录访问过的节点
    paths = []
    path = [start]
    sub_stack = []
    count = 0
    while len(path) != 0 and count < 5:
        if len(path) > deepth + 1:
            # 消栈
            path.pop()
            while len(sub_stack) != 0 and len(sub_stack[-1]) == 0:
                sub_stack.pop()
                path.pop()
            if len(sub_stack) != 0:
                sub_stack_top = sub_stack.pop()
                first = sub_stack_top[0]
                path.append(first)
                sub_stack_top.remove(first)
                sub_stack.append(sub_stack_top)
            continue
        node = path[-1]
        if node == end:
            count += 1
            # 栈顶就是终点
            # 1. 保存路径,主站弹出
            result = []
            for each in path:
                result.append(each)
            paths.append(result)
            # print(result)
            # 不能直接用这个，path是一个地址，这样path的值修改会影响到paths
            # paths.append(path)
            path.pop()
            # 2. 看辅栈是不是空的

            # 空的就退主栈和辅栈，直到不为空
            while len(sub_stack) != 0 and len(sub_stack[-1]) == 0:
                sub_stack.pop()
                path.pop()
            if len(sub_stack) != 0:
                sub_stack_top = sub_stack.pop()
                first = sub_stack_top[0]
                path.append(first)
                sub_stack_top.remove(first)
                sub_stack.append(sub_stack_top)
            continue
        temp = graph[node]
        nerbs = []
        # 过滤掉
        for each in temp:
            if each not in path:
                nerbs.append(each)

        if len(nerbs) != 0:
            first_nerb = nerbs[0]
            path.append(first_nerb)
            nerbs.remove(first_nerb)
            sub_stack.append(nerbs)
        else:
            # 邻居节点是空的，这条路走不通，主站退回去
            path.pop()
            while len(sub_stack) != 0 and len(sub_stack[-1]) == 0:
                sub_stack.pop()
                path.pop()
            if len(sub_stack) != 0:
                sub_stack_top = sub_stack.pop()
                first = sub_stack_top[0]
                path.append(first)
                sub_stack_top.remove(first)
                sub_stack.append(sub_stack_top)

    return paths


def trans(drug, path, name="drug", ):
    # 头实体都是drug
    if drug.startswith("DB"):
        entity_code = dbid2code[drug]
        # 获取embedding
        entity_vec = torch.tensor(drug_embedding[entity_code])
    else:
        entity_code = entityid2code[drug]
        entity_vec = torch.tensor(entity_embedding[entity_code])

    entity_vec = np.array(entity_vec / torch.sqrt(torch.sum(torch.pow(entity_vec, 2), dim=0)))

    for i in range(len(path) - 1):
        # 获取关系
        relation = graph_relation[path[i] + "-" + path[i + 1]]
        relation_vec = torch.tensor(relation_embedding[2 * relation])
        relation_vec = np.array(relation_vec / torch.sqrt(torch.sum(torch.pow(relation_vec, 2), dim=0)))
        entity_vec += relation_vec  # 张量相加

        # 获取节点
        end_node = path[i + 1]
        if end_node.startswith("DB"):
            end_node_code = dbid2code[end_node]
            # 获取embedding
            end_node_vec = torch.tensor(drug_embedding[end_node_code])
        else:
            end_node_code = entityid2code[end_node]
            end_node_vec = torch.tensor(entity_embedding[end_node_code])

        end_node_vec = np.array(end_node_vec / torch.sqrt(torch.sum(torch.pow(end_node_vec, 2), dim=0)))
        entity_vec += end_node_vec

    entity_vec = torch.tensor(entity_vec)
    entity_vec = np.array(entity_vec / torch.sqrt(torch.sum(torch.pow(entity_vec, 2), dim=0)))
    return entity_vec


def get_explaination(graph, start, end, file):
    deepth = 2
    nodes1 = get_nerbs(start, deepth, graph)
    sub_graph1 = get_subgraph(nodes1, graph)

    nodes2 = get_nerbs(end, deepth, graph)
    sub_graph2 = get_subgraph(nodes2, graph)
    nodes_mid = set(nodes1) & set(nodes2)
    paths = []
    results = []
    for node in nodes_mid:
        paths1 = findAllPath(sub_graph1, start, node, deepth=2)
        paths2 = findAllPath(sub_graph2, end, node, deepth=2)
        trans_vecs1 = []
        trans_vecs2 = []

        entity_code = entityid2code[node] if not node.startswith("DB") else dbid2code[node]
        mid_vec = torch.tensor(entity_embedding[entity_code])
        mid_vec = mid_vec / torch.sqrt(torch.sum(torch.pow(mid_vec, 2), dim=0))

        for path in paths1:
            trans_vecs1.append(trans(start, path))
        trans_vecs1 = torch.tensor(trans_vecs1)
        result1 = torch.matmul(trans_vecs1, mid_vec)

        for path in paths2:
            trans_vecs2.append(trans(end, path))
        trans_vecs2 = torch.tensor(trans_vecs2)
        result2 = torch.matmul(trans_vecs2, mid_vec)

        results.append([result1.max(), result2.max()])
        paths.append([paths1[result1.argmax()], paths2[result2.argmax()]])

    results = torch.tensor(results)
    logit = F.softmax(results[:, 0] * results[:, 1], dim=0)
    for each in logit.topk(10)[1]:
        path1 = list(paths[each][0])
        path2 = list(paths[each][1])
        path2.reverse()
        path = path1 + path2[1:]
        file.write(str(path))
        file.write("\n")


if __name__ == "__main__":
    drug_dis_topK_10 = {}
    with open("top15_disease.json", "r") as f:
        drug_dis_topK_10 = json.load(f)

    drug_list = []
    with open("full_not_final_have.json", "r") as f:
        drug_list = list(json.load(f).keys())
    drug_list.sort()

    file = open("paths_5.txt", "a", encoding="utf-8")
    selected = random.sample(drug_list, 30)
    print( selected)
    for drug in selected:
        diseases = drug_dis_topK_10[drug][0:10]
        for disease in diseases:

            try:
                if disease == "C0036572":
                    continue
                file.write("{} -- {}\n".format(drug, disease, ))
                print(drug, disease)
                get_explaination(graph, drug, disease, file)
                file.flush()
            except:
                print(drug, "error----------------")
                pass

    file.close()
