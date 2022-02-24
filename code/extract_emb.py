import numpy as np
import networkx as nx
import itertools
import random
import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score, recall_score, precision_score, matthews_corrcoef, confusion_matrix

def load_embedding(embedding_file_name, node_list=[]):
    R_node = []
    with open(embedding_file_name) as f:
        embedding_look_up = {}
        for line in f:
            vec = line.strip().split()
            #print(vec)
            node_id = vec[0]
            if node_id.startswith("I") or node_id.startswith("R"):
                if node_id not in node_list:
                    embeddings = vec[1:]
                    emb = [float(x) for x in embeddings]
                    emb = emb / np.linalg.norm(emb)
                    emb[np.isnan(emb)] = 0
                    embedding_look_up[node_id] = list(emb)
                    node_list.append(node_id)
                    if node_id.startswith("R"):
                        R_node.append(node_id)

    f.close()
    return embedding_look_up, R_node

def get_bench_circnode(filename):
    list = []
    with open(filename) as f:
        for node in f:
            node = node.strip('\n')
            #print(node)
            list.append(node)
    f.close()
    return list

def generate_bench_graph(filename):
    G = nx.read_edgelist(filename)
    return G

def generate_neg_edges(bench_graph, bench_node, R_node, num, seed):
    #print(bench_node)
    #print(R_node)
    G = nx.Graph()
    #print(G.nodes)
    G.add_nodes_from(bench_node)
    G.add_nodes_from(R_node)
    G.add_edges_from(itertools.product(bench_node, R_node))
    G.remove_edges_from(bench_graph.edges())
    #print(G.nodes)
    random.seed(seed)
    neg_edges = random.sample(G.edges, num)
    return neg_edges

def generate_pos_edges(bench_graph, num, seed):
    random.seed(seed)
    pos_edges = random.sample(bench_graph.edges, num)
    return pos_edges

def sigmod(x):
    return 1. / (1 + np.exp(-x))

def cosine(vec1, vec2):
    multiple = 0
    norm_vec1 = 0
    norm_vec2 = 0
    for v1, v2 in zip(vec1, vec2):
        multiple += v1 * v2
        norm_vec1 += v1 ** 2
        norm_vec2 += v2 ** 2
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0
    else:
        return multiple / ((norm_vec1 * norm_vec2) ** 0.5)

def evaluate_dot(pos_edges, neg_edges, embedding_look_up, dig=3, sig=False):
    y_pred_proba = []
    y_pred = []
    y_test = []
    if sig:
        for edge in pos_edges:
            if edge[0] in embedding_look_up.keys() and edge[1] in embedding_look_up.keys():
                node_u_emb = embedding_look_up[edge[0]]
                node_v_emb = embedding_look_up[edge[1]]
                emb_u = np.array(node_u_emb)
                emb_v = np.array(node_v_emb)
                score = np.dot(emb_u, emb_v)
                proba = sigmod(score)
                y_pred_proba.append(proba)
                if proba > 0.5:
                    y_pred.append(1)
                else:
                    y_pred.append(0)
                y_test.append(1)
        for edge in neg_edges:
            if edge[0] in embedding_look_up.keys() and edge[1] in embedding_look_up.keys():
                node_u_emb = embedding_look_up[edge[0]]
                node_v_emb = embedding_look_up[edge[1]]
                emb_u = np.array(node_u_emb)
                emb_v = np.array(node_v_emb)
                score = np.dot(emb_u, emb_v)
                proba = sigmod(score)
                y_pred_proba.append(proba)
                if proba > 0.5:
                    y_pred.append(1)
                else:
                    y_pred.append(0)
                y_test.append(0)
    else:
        for edge in pos_edges:
            if edge[0] in embedding_look_up.keys() and edge[1] in embedding_look_up.keys():
                node_u_emb = embedding_look_up[edge[0]]
                node_v_emb = embedding_look_up[edge[1]]
                emb_u = np.array(node_u_emb)
                emb_v = np.array(node_v_emb)
                score = np.dot(emb_u, emb_v)
                proba = score
                y_pred_proba.append(proba)
                if proba > 0:
                    y_pred.append(1)
                else:
                    y_pred.append(0)
                y_test.append(1)
        for edge in neg_edges:
            if edge[0] in embedding_look_up.keys() and edge[1] in embedding_look_up.keys():
                node_u_emb = embedding_look_up[edge[0]]
                node_v_emb = embedding_look_up[edge[1]]
                emb_u = np.array(node_u_emb)
                emb_v = np.array(node_v_emb)
                score = np.dot(emb_u, emb_v)
                proba = score
                y_pred_proba.append(proba)
                if proba > 0:
                    y_pred.append(1)
                else:
                    y_pred.append(0)
                y_test.append(0)
    print(y_test)
    print(y_pred_proba)
    print(y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    auc_pr = average_precision_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred,labels=[1,0])
    #pre = matrix[0][0] / (matrix[0][0]+matrix[1][0])
    #sen = matrix[0][0] / (matrix[0][0]+matrix[0][1])
    spe = matrix[1][1] / (matrix[1][1]+matrix[1][0])
    print('#' * 50)
    print(f'AUC-ROC: {auc_roc:.3f}, AUC-PR: {auc_pr:.3f}, Accuracy: {accuracy:.3f}, F1: {f1:.3f}, Recall/Sensitivity: {recall:.3f}, Precision: {precision:.3f}, MCC: {mcc:.3f}, Specificity: {spe:.3f}')
    print('#' * 50)
    # print(matrix)
    # print(pre)
    # print(sen)
    # print(spe)round(a,2)
    return round(auc_roc,dig), round(auc_pr,dig), round(accuracy,dig), round(f1,dig), round(recall,dig), round(precision,dig), round(mcc,dig), round(spe,dig)

def cal_fluctuate(score_list, num, dig=4):
    #print(score_list)
    #print(num)
    #num = len(score_list)
    #new = score_list.sort()
    #print(new)
    count = 0
    for s in score_list:
        count = count + s
    avg = count/num
    up = score_list[-1] - avg
    down = score_list[0] - avg
    if abs(up) > abs(down):
        flu = abs(up) * 0.1
    else:
        flu = abs(down) * 0.1

    return round(avg,dig), round(flu,dig)

def write_result(name):
    if name == 'our':
        path = "D:\hcx\\bibm2020\exp\\bibm\\fea\\embedding_our_layer=2_onlytrain.txt"
    else:
        path = "D:\hcx\\bibm2020\exp\\bibm\\connect\\vec_con_" + name + ".txt"
    bench_node = 'D:\hcx\\bibm2020\exp\\bibm\\connect\\bench_circ_node.txt'
    bench_path = 'D:\hcx\\bibm2020\exp\\bibm\\connect\\bench.txt'
    output = open('D:\hcx\\bibm2020\exp\\bibm\\connect\\result.txt', 'a', newline='')

    # t1 = time.time()
    AUCR_list = []
    AUCP_list = []
    Acc_list = []
    F1_list = []
    Recall_list = []
    Pre_list = []
    # embedding = {}
    # Rnode = []
    # benchnode = []
    embedding, Rnode = load_embedding(path)
    benchnode = get_bench_circnode(bench_node)
    benchgraph = generate_bench_graph(bench_path)
    for i in range(100):
        pos = generate_pos_edges(benchgraph, 1000, i)
        neg = generate_neg_edges(benchgraph, benchnode, Rnode, 200, i)
        # print(neg)
        a, b, c, d, e, f, g, h = evaluate_dot(pos, neg, embedding, dig=4, sig=False)
        AUCR_list.append(a)
        AUCP_list.append(b)
        Acc_list.append(c)
        F1_list.append(d)
        Recall_list.append(e)
        Pre_list.append(f)
        # benchgraph.clear()

    # output.write(str(a)+' '+str(b)+' '+str(c)+' '+str(d)+' '+str(e)+' '+str(f)+' '+str(g)+' '+str(h)+'\n')
    # evaluate_cosine(pos,neg,embedding)
    # print(AUCR_list)
    AUCR_list.sort()
    AUCP_list.sort()
    Acc_list.sort()
    F1_list.sort()
    Recall_list.sort()
    Pre_list.sort()
    # print(AUCR_list)
    AUCR_avg, AUCR_flu = cal_fluctuate(AUCR_list, len(AUCR_list))
    AUCP_avg, AUCP_flu = cal_fluctuate(AUCP_list, len(AUCP_list))
    Acc_avg, Acc_flu = cal_fluctuate(Acc_list, len(Acc_list))
    F1_avg, F1_flu = cal_fluctuate(F1_list, len(F1_list))
    Recall_avg, Recall_flu = cal_fluctuate(Recall_list, len(Recall_list))
    Pre_avg, Pre_flu = cal_fluctuate(Pre_list, len(Pre_list))
    output.write(name+'_600_2 '+',')
    output.write(str(AUCR_avg) + ' ' + str(AUCR_flu) + ',')
    output.write(str(AUCP_avg) + ' ' + str(AUCP_flu) + ',')
    output.write(str(Acc_avg) + ' ' + str(Acc_flu) + ',')
    output.write(str(F1_avg) + ' ' + str(F1_flu) + ',')
    output.write(str(Recall_avg) + ' ' + str(Recall_flu) + ',')
    output.write(str(Pre_avg) + ' ' + str(Pre_flu) + '\n')
    #benchgraph.clear()


# def evaluate_cosine(pos_edges, neg_edges, embedding_look_up):
#     y_pred_proba = []
#     y_pred = []
#     y_test = []
#     for edge in pos_edges:
#         if edge[0] in embedding_look_up.keys() and edge[1] in embedding_look_up.keys():
#             node_u_emb = embedding_look_up[edge[0]]
#             node_v_emb = embedding_look_up[edge[1]]
#             score = cosine(node_u_emb, node_v_emb)
#             proba = score
#             y_pred_proba.append(proba)
#             if proba > 0:
#                 y_pred.append(1)
#             else:
#                 y_pred.append(0)
#             y_test.append(1)
#     for edge in neg_edges:
#         if edge[0] in embedding_look_up.keys() and edge[1] in embedding_look_up.keys():
#             node_u_emb = embedding_look_up[edge[0]]
#             node_v_emb = embedding_look_up[edge[1]]
#             score = cosine(node_u_emb, node_v_emb)
#             proba = score
#             y_pred_proba.append(proba)
#             if proba > 0:
#                 y_pred.append(1)
#             else:
#                 y_pred.append(0)
#             y_test.append(0)
#     print(y_test)
#     print(y_pred_proba)
#     print(y_pred)
#     auc_roc = roc_auc_score(y_test, y_pred_proba)
#     auc_pr = average_precision_score(y_test, y_pred_proba)
#     accuracy = accuracy_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
#     print('#' * 50)
#     print(f'AUC-ROC: {auc_roc:.3f}, AUC-PR: {auc_pr:.3f}, Accuracy: {accuracy:.3f}, F1: {f1:.3f}, Recall: {recall:.3f}, Precision: {precision:.3f}')
#     print('#' * 50)
#     return auc_roc, auc_pr, accuracy, f1, recall, precision

if __name__ == "__main__":
    method = ['eden']
    t1 = time.time()
    write_result(method[10])
    t2 = time.time()
    print("time:"+str(t2-t1))


    # plt.figure()
    # lw = 2
    # plt.figure(figsize=(10, 10))
    # plt.plot(fpr, tpr, color='darkorange',
    #          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    # plt.legend(loc="lower right")
    # plt.show()








