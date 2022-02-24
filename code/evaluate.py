import numpy as np
import networkx as nx
import itertools
import random
import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score, recall_score, precision_score, matthews_corrcoef, confusion_matrix


def sigmod(x):
    return 1. / (1 + np.exp(-x))

def load_embedding(embedding_file_name,norm=False):
    with open(embedding_file_name) as f:
        embedding_look_up = {}
        for line in f:
            vec = line.strip().split()
            #print(vec)
            node_id = vec[0]
            embeddings = vec[1:]
            emb = [float(x) for x in embeddings]
            if norm:
                emb = emb / np.linalg.norm(emb)
                emb[np.isnan(emb)] = 0
            embedding_look_up[node_id] = list(emb)

    f.close()
    return embedding_look_up

def generate_bench_graph(filename):
    G = nx.read_edgelist(filename)
    return G

def generate_neg_edges(neg_filename):
    G_neg = nx.read_edgelist(neg_filename)
    neg_edges =  G_neg.edges
    # random.seed(seed)
    # neg_edges = random.sample(G_neg.edges, num)
    return neg_edges

def generate_pos_edges(pos_filename):
    G_pos = nx.read_edgelist(pos_filename)
    pos_edges = G_pos.edges
    # random.seed(seed)
    # pos_edges = random.sample(G_pos.edges, num)
    return pos_edges

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
                # print(edge[0])
                # print(emb_u)
                # print(edge[1])
                # print(emb_v)
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
        flu = abs(up)
    else:
        flu = abs(down)

    return round(avg,dig), round(flu,dig)

def write_result(name):
    # if name.startswith('e'):
    #     path = r'D:\hcx\bibm2020\data\new_train\exp\our_embedding\\'+name+'.txt'
    # else:
    #     path = r'D:\hcx\bibm2020\data\new_train\exp\vec_newt_' + name + ".txt"
    path = r'D:\hcx\bibm2020-ex\exp\ablation\\'+name+'.csv'
    pos_path = r'D:\hcx\bibm2020\parameter\test_data\pos.txt'
    output = open(r'D:\hcx\bibm2020-ex\exp\ablation\result_new.txt', 'a', newline='')

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
    print(path)
    embedding = load_embedding(path, norm=False)
    for i in range(100):
        neg_path = r'D:\hcx\bibm2020\parameter\test_data\neg_'+str(i)+'.txt'
        pos_edges = generate_pos_edges(pos_path)
        neg_edges = generate_neg_edges(neg_path)
        # print(neg)
        a, b, c, d, e, f, g, h = evaluate_dot(pos_edges, neg_edges, embedding, dig=4, sig=False)
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
    output.write('out_dim '+name+',')
    output.write(str(AUCR_avg) + ' ' + str(AUCR_flu) + ',')
    output.write(str(AUCP_avg) + ' ' + str(AUCP_flu) + ',')
    output.write(str(Acc_avg) + ' ' + str(Acc_flu) + ',')
    output.write(str(F1_avg) + ' ' + str(F1_flu) + ',')
    output.write(str(Recall_avg) + ' ' + str(Recall_flu) + ',')
    output.write(str(Pre_avg) + ' ' + str(Pre_flu) + '\n')

if __name__ == "__main__":
    method = ['eden']
    our = ['embedding_800_gat','embedding_800_gcn','embedding_800_sage']
    write_result('embedding_cmgd')


