import EDEN.utility.metrics as metrics
from EDEN.utility.parser import parse_args
import multiprocessing
import heapq
import numpy as np

from EDEN.utility.loader import EDEN_loader

cores = multiprocessing.cpu_count() // 2

args = parse_args()
Ks = eval(args.Ks)

data_generator = EDEN_loader(args=args, path=args.data_path + args.dataset)
batch_test_flag = False

DRUG_NUM, DISEASE_NUM = data_generator.n_drugs, data_generator.n_diseases
N_TRAIN, N_TEST = data_generator.n_train, data_generator.n_test
BATCH_SIZE = args.batch_size

def ranklist_by_heapq(drug_pos_test, test_diseases, rating, Ks):
    disease_score = {}
    for i in test_diseases:
        disease_score[i] = rating[i]

    K_max = max(Ks)
    K_max_disease_score = heapq.nlargest(K_max, disease_score, key=disease_score.get)

    r = []
    for i in K_max_disease_score:
        if i in drug_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    return r, auc

def get_auc(disease_score, drug_pos_test):
    disease_score = sorted(disease_score.diseases(), key=lambda kv: kv[1])
    disease_score.reverse()
    disease_sort = [x[0] for x in disease_score]
    posterior = [x[1] for x in disease_score]

    r = []
    for i in disease_sort:
        if i in drug_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = metrics.auc(ground_truth=r, prediction=posterior)
    return auc

def ranklist_by_sorted(drug_pos_test, test_diseases, rating, Ks):
    disease_score = {}
    for i in test_diseases:
        disease_score[i] = rating[i]

    K_max = max(Ks)
    K_max_disease_score = heapq.nlargest(K_max, disease_score, key=disease_score.get)

    r = []
    for i in K_max_disease_score:
        if i in drug_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(disease_score, drug_pos_test)
    return r, auc


def get_performance(drug_pos_test, r, auc, Ks):
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks:
        precision.append(metrics.precision_at_k(r, K))
        recall.append(metrics.recall_at_k(r, K, len(drug_pos_test)))
        ndcg.append(metrics.ndcg_at_k(r, K))
        hit_ratio.append(metrics.hit_at_k(r, K))

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}


def test_one_drug(x):
    # drug u's ratings for drug u
    rating = x[0]
    #uid
    u = x[1]
    #drug u's diseases in the training set
    try:
        training_diseases = data_generator.train_drug_dict[u]
    except Exception:
        training_diseases = []
    #drug u's diseases in the test set
    drug_pos_test = data_generator.test_drug_dict[u]

    all_diseases = set(range(DISEASE_NUM))

    test_diseases = list(all_diseases - set(training_diseases))

    if args.test_flag == 'part':
        r, auc = ranklist_by_heapq(drug_pos_test, test_diseases, rating, Ks)
    else:
        r, auc = ranklist_by_sorted(drug_pos_test, test_diseases, rating, Ks)


    return get_performance(drug_pos_test, r, auc, Ks)


def test(sess, model, drugs_to_test, drop_flag=False, batch_test_flag=False):
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}

    pool = multiprocessing.Pool(cores)

    if args.model_type in ['ripple']:

        u_batch_size = BATCH_SIZE
        i_batch_size = BATCH_SIZE // 20
    elif args.model_type in ['fm', 'nfm']:
        u_batch_size = BATCH_SIZE
        i_batch_size = BATCH_SIZE
    else:
        u_batch_size = BATCH_SIZE * 2
        i_batch_size = BATCH_SIZE

    test_drugs = drugs_to_test
    n_test_drugs = len(test_drugs)
    n_drug_batchs = n_test_drugs // u_batch_size + 1

    count = 0

    for u_batch_id in range(n_drug_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        drug_batch = test_drugs[start: end]

        if batch_test_flag:

            n_disease_batchs = DISEASE_NUM // i_batch_size + 1
            rate_batch = np.zeros(shape=(len(drug_batch), DISEASE_NUM))

            i_count = 0
            for i_batch_id in range(n_disease_batchs):
                i_start = i_batch_id * i_batch_size
                i_end = min((i_batch_id + 1) * i_batch_size, DISEASE_NUM)

                disease_batch = range(i_start, i_end)

                feed_dict = data_generator.generate_test_feed_dict(model=model,
                                                                   drug_batch=drug_batch,
                                                                   disease_batch=disease_batch,
                                                                   drop_flag=drop_flag)
                i_rate_batch = model.eval(sess, feed_dict=feed_dict)
                i_rate_batch = i_rate_batch.reshape((-1, len(disease_batch)))

                rate_batch[:, i_start: i_end] = i_rate_batch
                i_count += i_rate_batch.shape[1]

            assert i_count == DISEASE_NUM

        else:
            disease_batch = range(DISEASE_NUM)
            feed_dict = data_generator.generate_test_feed_dict(model=model,
                                                               drug_batch=drug_batch,
                                                               disease_batch=disease_batch,
                                                               drop_flag=drop_flag)
            rate_batch = model.eval(sess, feed_dict=feed_dict)
            rate_batch = rate_batch.reshape((-1, len(disease_batch)))

        drug_batch_rating_uid = zip(rate_batch, drug_batch)
        batch_result = pool.map(test_one_drug, drug_batch_rating_uid)
        count += len(batch_result)

        for re in batch_result:
            result['precision'] += re['precision']/n_test_drugs
            result['recall'] += re['recall']/n_test_drugs
            result['ndcg'] += re['ndcg']/n_test_drugs
            result['hit_ratio'] += re['hit_ratio']/n_test_drugs
            result['auc'] += re['auc']/n_test_drugs


    assert count == n_test_drugs
    pool.close()
    return result