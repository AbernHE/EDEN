import pandas as pd
import random
import json
import pickle as pkl
import csv
import numpy as np
import math

dbid2code = {}
with open("dbid2code.json", "r") as f:
    dbid2code = json.load(f)

entityid2code = {}
with open("entityid2code.json", "r") as f:
    entityid2code = json.load(f)

train_part = 0.6
part = 6
part2 = 4

# drug_pro
# data_drug_pro = pd.read_csv('drug_pro.csv', usecols=[0, 2], names=["drugId", "diseaseId"], header=None,
#                             skiprows=1)  # train
# print(data_drug_pro.head())
# data_drug_pro.drop_duplicates(inplace=True)
data_drug_dis_final = pd.read_csv('drug_dis_final.csv', usecols=["drugId", "diseaseId"])
data_drug_dis_full = pd.read_csv('drug_dis_full.csv')
data_drug_dis_full = data_drug_dis_full[data_drug_dis_full.relation == "Approved"]
data_drug_dis_full.drop("relation", inplace=True, axis=1)
print(data_drug_dis_full.head())
data_drug_dis_final.drop_duplicates(inplace=True)
data_drug_dis_full.drop_duplicates(inplace=True)
data_drug_dis = data_drug_dis_final.append(data_drug_dis_full, ignore_index=True)
data_drug_dis.drop_duplicates(subset=["drugId", "diseaseId"], inplace=True)
# data = data_drug_dis.append(data_drug_pro,ignore_index=True)


data_dict = {}
for _, row in data_drug_dis.iterrows():
    # for _, row in data.iterrows():
    drug = str(dbid2code[row[0]])
    entity = str(entityid2code[row[1]])
    if drug not in data_dict:
        data_dict[drug] = []
    data_dict[drug].append(entity)

#
# file_train = open("nozero_train_mixed_{}{}.txt".format(part,part2), "w", encoding="utf8")
# file_test = open("nozero_test_mixed_{}{}.txt".format(part,part2), "w", encoding="utf8")

file_train = open("nozero_train_wio_pro_{}{}.txt".format(part, part2), "w", encoding="utf8")
file_test = open("nozero_test_wio_pro_{}{}.txt".format(part, part2), "w", encoding="utf8")

for drug in data_dict:
    file_train.write(str(drug))
    relations = data_dict[drug]
    total_number = len(relations)
    train_indices = random.sample([x for x in range(total_number)], math.ceil(train_part * total_number))
    # train
    for i in train_indices:
        file_train.write(" " + str(relations[i]))
    file_train.write("\n")

    if len(train_indices) < total_number:
        test_indices = []
        for i in range(total_number):
            if i not in train_indices:
                test_indices.append(i)

        file_test.write(str(drug))
        for i in test_indices:
            file_test.write(" " + str(relations[i]))
        file_test.write("\n")

file_train.close()
file_test.close()
