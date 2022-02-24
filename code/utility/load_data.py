import collections
import numpy as np
import random as rd

class Data(object):
    def __init__(self, args, path):
        self.path = path
        self.args = args

        self.batch_size = args.batch_size

        train_file = path + '/train.txt'
        test_file = path + '/test.txt'

        DIN_file = path + '/final.txt'

        # ----------get number of drugs and diseases & then load rating data from train_file & test_file------------.
        self.n_train, self.n_test = 0, 0
        self.n_drugs, self.n_diseases = 0, 0

        self.train_data, self.train_drug_dict = self._load_ratings(train_file)
        self.test_data, self.test_drug_dict = self._load_ratings(test_file)
        self.exist_drugs = self.train_drug_dict.keys()

        self._statistic_ratings()

        # ----------get number of entities and relations & then load DIN data from DIN_file ------------.
        self.n_relations, self.n_entities, self.n_triples = 0, 0, 0
        self.DIN_data, self.DIN_dict, self.relation_dict = self._load_DIN(DIN_file)

        # ----------print the basic info about the dataset-------------.
        self.batch_size_DIN = args.batch_size_DIN
        self._print_data_info()

    # reading train & test interaction data.
    def _load_ratings(self, file_name):
        drug_dict = dict()
        inter_mat = list()

        lines = open(file_name, 'r').readlines()
        for l in lines:
            tmps = l.strip()
            inters = [int(i) for i in tmps.split(' ')]

            u_id, pos_ids = inters[0], inters[1:]
            pos_ids = list(set(pos_ids))

            for i_id in pos_ids:
                inter_mat.append([u_id, i_id])

            if len(pos_ids) > 0:
                drug_dict[u_id] = pos_ids
        return np.array(inter_mat), drug_dict

    def _statistic_ratings(self):
        self.n_drugs = max(max(self.train_data[:, 0]), max(self.test_data[:, 0])) + 1
        self.n_diseases = max(max(self.train_data[:, 1]), max(self.test_data[:, 1])) + 1
        self.n_train = len(self.train_data)
        self.n_test = len(self.test_data)

    # reading train & test interaction data.
    def _load_DIN(self, file_name):
        def _construct_DIN(DIN_np):
            DIN = collections.defaultdict(list)
            rd = collections.defaultdict(list)

            for head, relation, tail in DIN_np:
                DIN[head].append((tail, relation))
                rd[relation].append((head, tail))
            return DIN, rd

        DIN_np = np.loadtxt(file_name, dtype=np.int32)
        DIN_np = np.unique(DIN_np, axis=0)

        # self.n_relations = len(set(DIN_np[:, 1]))
        # self.n_entities = len(set(DIN_np[:, 0]) | set(DIN_np[:, 2]))
        self.n_relations = max(DIN_np[:, 1]) + 1
        self.n_entities = max(max(DIN_np[:, 0]), max(DIN_np[:, 2])) + 1
        self.n_triples = len(DIN_np)

        DIN_dict, relation_dict = _construct_DIN(DIN_np)

        return DIN_np, DIN_dict, relation_dict

    def _print_data_info(self):
        print('[n_drugs, n_diseases]=[%d, %d]' % (self.n_drugs, self.n_diseases))
        print('[n_train, n_test]=[%d, %d]' % (self.n_train, self.n_test))
        print('[n_entities, n_relations, n_triples]=[%d, %d, %d]' % (self.n_entities, self.n_relations, self.n_triples))
        print('[batch_size, batch_size_DIN]=[%d, %d]' % (self.batch_size, self.batch_size_DIN))

    def _generate_train_cf_batch(self):
        if self.batch_size <= self.n_drugs:
            drugs = rd.sample(self.exist_drugs, self.batch_size)
        else:
            drugs = [rd.choice(self.exist_drugs) for _ in range(self.batch_size)]

        def sample_pos_diseases_for_u(u, num):
            pos_diseases = self.train_drug_dict[u]
            n_pos_diseases = len(pos_diseases)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_diseases, size=1)[0]
                pos_i_id = pos_diseases[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_diseases_for_u(u, num):
            neg_diseases = []
            while True:
                if len(neg_diseases) == num: break
                neg_i_id = np.random.randint(low=0, high=self.n_diseases,size=1)[0]

                if neg_i_id not in self.train_drug_dict[u] and neg_i_id not in neg_diseases:

                    neg_diseases.append(neg_i_id)


            return neg_diseases

        pos_diseases, neg_diseases = [], []
        for u in drugs:
            pos_diseases += sample_pos_diseases_for_u(u, 1)
            neg_diseases += sample_neg_diseases_for_u(u, 1)

        return drugs, pos_diseases, neg_diseases

    def get_sparsity_split(self):
        try:
            split_uids, split_state = [], []
            lines = open(self.path + '/sparsity.split', 'r').readlines()

            for idx, line in enumerate(lines):
                if idx % 2 == 0:
                    split_state.append(line.strip())
                    print(line.strip())
                else:
                    split_uids.append([int(uid) for uid in line.strip().split(' ')])
            print('get sparsity split.')

        except Exception:
            split_uids, split_state = self.create_sparsity_split()
            f = open(self.path + '/sparsity.split', 'w')
            for idx in range(len(split_state)):
                f.write(split_state[idx] + '\n')
                f.write(' '.join([str(uid) for uid in split_uids[idx]]) + '\n')
            print('create sparsity split.')

        return split_uids, split_state



    def create_sparsity_split(self):
        all_drugs_to_test = list(self.test_drug_dict.keys())
        drug_n_iid = dict()

        # generate a dictionary to store (key=n_iids, value=a list of uid).
        for uid in all_drugs_to_test:
            train_iids = self.train_drug_dict[uid]
            test_iids = self.test_drug_dict[uid]

            n_iids = len(train_iids) + len(test_iids)

            if n_iids not in drug_n_iid.keys():
                drug_n_iid[n_iids] = [uid]
            else:
                drug_n_iid[n_iids].append(uid)
        split_uids = list()

        # split the whole drug set into four subset.
        temp = []
        count = 1
        fold = 4
        n_count = (self.n_train + self.n_test)
        n_rates = 0

        split_state = []
        for idx, n_iids in enumerate(sorted(drug_n_iid)):
            temp += drug_n_iid[n_iids]
            n_rates += n_iids * len(drug_n_iid[n_iids])
            n_count -= n_iids * len(drug_n_iid[n_iids])

            if n_rates >= count * 0.25 * (self.n_train + self.n_test):
                split_uids.append(temp)

                state = '#inter per drug<=[%d], #drugs=[%d], #all rates=[%d]' %(n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

                temp = []
                n_rates = 0
                fold -= 1

            if idx == len(drug_n_iid.keys()) - 1 or n_count == 0:
                split_uids.append(temp)

                state = '#inter per drug<=[%d], #drugs=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)


        return split_uids, split_state