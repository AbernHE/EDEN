import torch
import pickle as pkl
import json
import random
import numpy as np
import torch.nn.functional as F
import time

dbid2code = {}
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
with open("graph_fenlei.json", "rb") as f:
    graph = json.load(f)

graph_relation = {}
with open("graph_relation.json", "r") as f:
    graph_relation = json.load(f)


def get_nerbs(entity, graph, path_type=["dis_gene", "gene_pathway"]):
    n1 = [entity]
    visited = []
    for each in path_type:
        n2 = []
        for node in n1:
            visited.append(node)
            try:
                temp = graph[node][each]
                for nerb in temp:
                    if nerb not in visited:
                        n2.append(nerb)
            except:
                pass
        n1 = n2
    return n2


def findAllPath(graph, start, end, end_vec, path_type=["dis_gene", ], ):
    # 用一个栈记录中间节点，
    # 用一个list记录访问过的节点
    if start.startswith("DB") or start.startswith("MESH"):
        start_code = dbid2code[start]
        # 获取embedding
        start_vec = drug_embedding[start_code]
    else:
        start_code = entityid2code[start]
        start_vec = entity_embedding[start_code]

    start_vec = torch.tensor(start_vec)
    path = [start, ]

    score = [torch.matmul(start_vec, end_vec)]
    sub_stack = []
    while len(path) != 0:
        node = path[-1]
        if node == end:
            return path, score

        if len(path) == len(path_type) + 1:
            # 消栈
            path.pop()
            score.pop()
            while len(sub_stack) != 0 and len(sub_stack[-1]) == 0:
                sub_stack.pop()
                path.pop()
                score.pop()
            if len(sub_stack) != 0:
                sub_stack_top = sub_stack.pop()
                first = sub_stack_top[0]
                path.append(first[0])
                score.append(first[1])
                sub_stack_top.remove(first)
                sub_stack.append(sub_stack_top)
            continue
        try:
            temp = graph[node][path_type[len(path) - 1]]
        except:
            temp = []
        nerbs = []
        # 过滤掉
        for each in temp:
            if each not in path:
                nerbs.append(each)

        if len(nerbs) != 0:
            entitiy_vecs = []
            # 算分
            for each in nerbs:
                if each.startswith("DB") or each.startswith("MESH"):
                    entity_code = dbid2code[each]
                    # 获取embedding
                    entity_vec = drug_embedding[entity_code]
                else:
                    entity_code = entityid2code[each]
                    entity_vec = entity_embedding[entity_code]
                entitiy_vecs.append(entity_vec)

            entitiy_vecs = torch.tensor(entitiy_vecs)
            result = torch.matmul(entitiy_vecs, end_vec)
            result_ = [(nerbs[i], result[i]) for i in range(len(result))]
            result_.sort(key=lambda x: x[1])
            first = result_[0]
            path.append(first[0])
            score.append(first[1])
            result_.remove(first)
            sub_stack.append(result_)
        else:
            # 邻居节点是空的，这条路走不通，主站退回去
            path.pop()
            score.pop()
            while len(sub_stack) != 0 and len(sub_stack[-1]) == 0:
                sub_stack.pop()
                path.pop()
                score.pop()
            if len(sub_stack) != 0:
                sub_stack_top = sub_stack.pop()
                first = sub_stack_top[0]
                path.append(first[0])
                score.append(first[1])
                sub_stack_top.remove(first)
                sub_stack.append(sub_stack_top)

    return None, None


def get_explaination(graph, start, end, file, path_type1, path_type2):
    nodes1 = get_nerbs(start, graph, path_type1)
    nodes2 = get_nerbs(end, graph, path_type2)
    nodes = set(nodes1) & set(nodes2)
    print(nodes)

    paths = []
    scores1 = []
    scores2 = []
    for node in nodes:
        entity_code = entityid2code[node]
        mid_vec = torch.tensor(entity_embedding[entity_code])
        mid_vec = mid_vec / torch.sqrt(torch.sum(torch.pow(mid_vec, 2), dim=0))

        path1, score1 = findAllPath(graph, start, node, mid_vec, path_type1)
        path2, score2 = findAllPath(graph, end, node, mid_vec, path_type2)

        if score1 == None or score2 == None:
            continue

        score = 0
        for each in score1:
            score += each.item()
        scores1.append(score)

        score = 0
        for each in score2:
            score += each.item()
        scores2.append(score)

        paths.append([path1, path2])

    scores1 = torch.tensor(scores1)
    scores2 = torch.tensor(scores2)
    result = scores1 * scores2
    if result.shape[0] != 0:
        result = result / torch.sqrt(torch.sum(torch.pow(result, 2), dim=0))
        logit = F.softmax(result, dim=0)
        num = logit.shape[0]
        K = 10
        for each in logit.topk(min(num, K))[1]:
            path1 = list(paths[each][0])
            path2 = list(paths[each][1])
            path2.reverse()
            path = path1 + path2[1:]
            file.write(str(path))
            file.write("\n")


if __name__ == "__main__":
    # path_type = ["drug_dis", "dis_gene", "gene_pathway"]
    # path_type1 = ["drug_dis", "dis_gene", "gene_pathway"]
    # path_type2 = [ "dis_gene", "gene_pathway"]
    # path_type1 = ["drug_pro", "pro_pro"]
    # path_type2 = ["dis_gene","gene_pro"]
    path_type1 = ["drug_pro", "pro_gene","gene_go"]
    path_type2 = ["dis_gene", "gene_go"]

    drug_dis_topK_10 = {}
    with open("top15_disease.json", "r") as f:
        drug_dis_topK_10 = json.load(f)

    drug_list = []
    with open("full_not_final_have.json", "r") as f:
        drug_list = list(json.load(f).keys())
    drug_list.sort()

    file = open("paths_drug_pro_g_go_g_dis.txt", "a", encoding="utf-8")
    selected = random.sample(drug_list, 30)
    print("paths_drug_pro_g_go_g_dis",selected)
    for drug in selected:
        diseases = drug_dis_topK_10[drug][0:10]
        for disease in diseases:
            file.write("{} -- {}\n".format(drug, disease, ))
            print(drug, disease)
            get_explaination(graph, drug, disease, file, path_type1, path_type2)
            file.flush()

    file.close()
