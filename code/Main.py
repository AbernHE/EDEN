import argparse
import tensorflow as tf
from EDEN.utility.helper import *
from EDEN.utility.batch_test import *
from EDEN.utility.parser import parser
from time import time
import numpy as np

from EDEN.model import model

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

if __name__ == '__main__':
    tf.set_random_seed(2019)
    np.random.seed(2019)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    config = dict()
    config['n_drugs'] = data_generator.n_drugs
    config['n_diseases'] = data_generator.n_diseases
    config['n_relations'] = data_generator.n_relations
    config['n_entities'] = data_generator.n_entities
    config['A_in'] = sum(data_generator.lap_list)
    config['all_h_list'] = data_generator.all_h_list
    config['all_r_list'] = data_generator.all_r_list
    config['all_t_list'] = data_generator.all_t_list
    config['all_v_list'] = data_generator.all_v_list

    t0 = time()

    model = model(data_config=config, pretrain_data=None, args=args)
    saver = tf.train.Saver()
    if args.save_flag == 1:
        layer = '-'.join([str(l) for l in eval(args.layer_size)])
        weights_save_path = '%sweights/%s/%s/%s/l%s_r%s' % (
            args.weights_path, args.dataset, model.model_type, layer, str(args.lr),
            '-'.join([str(r) for r in eval(args.regs)]))

        ensureDir(weights_save_path)
        save_saver = tf.train.Saver(max_to_keep=1)

     config = tf.ConfigProto()
     config.gpu_options.allow_growth = True
     sess = tf.Session(config=config)

if args.report != 1:
    drugs_to_test = list(data_generator.test_drug_dict.keys())

    ret = test(sess, model, drugs_to_test, drop_flag=False, batch_test_flag=batch_test_flag)
    cur_best_pre_0 = ret['recall'][0]

    pretrain_ret = 'pretrained model recall=[%.5f, %.5f], precision=[%.5f, %.5f], hit=[%.5f, %.5f],' \
                   'ndcg=[%.5f, %.5f], auc=[%.5f]' % \
                   (ret['recall'][0], ret['recall'][-1],
                    ret['precision'][0], ret['precision'][-1],
                    ret['hit_ratio'][0], ret['hit_ratio'][-1],
                    ret['ndcg'][0], ret['ndcg'][-1], ret['auc'])
    print(pretrain_ret)

   if args.save_flag == -1:
        drug_embed, disease_embed = sess.run(
            [model.weights['drug_embedding'], model.weights['disease_embedding']],
            feed_dict={})
        # temp_save_path = '%spretrain/%s/%s/%s_%s.npz' % (args.proj_path, args.dataset, args.model_type, str(args.lr),
        #                                                  '-'.join([str(r) for r in eval(args.regs)]))
        temp_save_path = '%spretrain/%s/%s.npz' % (args.proj_path, args.dataset, model.model_type)
        ensureDir(temp_save_path)
        np.savez(temp_save_path, drug_embed=drug_embed, disease_embed=disease_embed)
        print('save the weights of fm in path: ', temp_save_path)
        exit()

    if args.save_flag == -2:
        drug_embed, entity_embed, relation_embed = sess.run(
            [model.weights['drug_embed'], model.weights['entity_embed'], model.weights['relation_embed']],
            feed_dict={})

        temp_save_path = '%spretrain/%s/%s.npz' % (args.proj_path, args.dataset, args.model_type)
        ensureDir(temp_save_path)
        np.savez(temp_save_path, drug_embed=drug_embed, entity_embed=entity_embed, relation_embed=relation_embed)
        print('save the weights of model in path: ', temp_save_path)
        exit()

     else:
        sess.run(tf.global_variables_initializer())
        cur_best_pre_0 = 0.
        print('without pretraining.')

    if args.report == 1:
        assert args.test_flag == 'full'
        drugs_to_test_list, split_state = data_generator.get_sparsity_split()

        drugs_to_test_list.append(list(data_generator.test_drug_dict.keys()))
        split_state.append('all')

        save_path = '%sreport/%s/%s.result' % (args.proj_path, args.dataset, model.model_type)
        ensureDir(save_path)
        f = open(save_path, 'w')
        f.write('embed_size=%d, lr=%.4f, regs=%s, loss_type=%s, \n' % (args.embed_size, args.lr, args.regs,
                                                                       args.loss_type))

        for i, drugs_to_test in enumerate(drugs_to_test_list):
            ret = test(sess, model, drugs_to_test, drop_flag=False, batch_test_flag=batch_test_flag)

            final_perf = "recall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                         ('\t'.join(['%.5f' % r for r in ret['recall']]),
                          '\t'.join(['%.5f' % r for r in ret['precision']]),
                          '\t'.join(['%.5f' % r for r in ret['hit_ratio']]),
                          '\t'.join(['%.5f' % r for r in ret['ndcg']]))
            print(final_perf)

            f.write('\t%s\n\t%s\n' % (split_state[i], final_perf))
        f.close()
        exit()

    """
    *********************************************************
    Train.
    """
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    stopping_step = 0
    should_stop = False

    for epoch in range(args.epoch):
        t1 = time()
        loss, base_loss, DIN_loss, reg_loss = 0., 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1

        """
        *********************************************************
        Alternative Training for model:
        ... phase 1
        """
        for idx in range(n_batch):
            btime= time()

            batch_data = data_generator.generate_train_batch()
            feed_dict = data_generator.generate_train_feed_dict(model, batch_data)

            _, batch_loss, batch_base_loss, batch_DIN_loss, batch_reg_loss = model.train(sess, feed_dict=feed_dict)

            loss += batch_loss
            base_loss += batch_base_loss
            DIN_loss += batch_DIN_loss
            reg_loss += batch_reg_loss

        if np.isnan(loss) == True:
            print('ERROR: loss@phase1 is nan.')
            sys.exit()

        """
        *********************************************************
        Alternative Training for model:
        ... phase 2
        """
        if args.model_type in ['l2']:

            n_A_batch = len(data_generator.all_h_list) // args.batch_size_DIN + 1

            if args.use_DIN is True:
                for idx in range(n_A_batch):
                    btime = time()

                    A_batch_data = data_generator.generate_train_A_batch()
                    feed_dict = data_generator.generate_train_A_feed_dict(model, A_batch_data)
                    print('debug-2')
                    print(feed_dict)
                    print(len(feed_dict[model.h]))
                    print(len(feed_dict[model.r]))
                    print(len(feed_dict[model.pos_t]))
                    print(len(feed_dict[model.neg_t]))

                    _, batch_loss, batch_DIN_loss, batch_reg_loss, mutu_loss = model.train_A(sess, feed_dict=feed_dict)

                    loss += batch_loss
                    DIN_loss += batch_DIN_loss
                    DIN_loss += mutu_loss
                    reg_loss += batch_reg_loss

            if args.use_att is True:
                # updating attentive laplacian matrix.
                model.update_attentive_A(sess)

        if np.isnan(loss) == True:
            print('ERROR: loss@phase2 is nan.')
            sys.exit()

        show_step = 10
        if (epoch + 1) % show_step != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f]' % (
                    epoch, time() - t1, loss, base_loss, DIN_loss, reg_loss)
                print(perf_str)
            continue

        """
        *********************************************************
        Test.
        """
        t2 = time()
        drugs_to_test = list(data_generator.test_drug_dict.keys())

        ret = test(sess, model, drugs_to_test, drop_flag=False, batch_test_flag=batch_test_flag)

        """
        *********************************************************
        Performance logging.
        """
        t3 = time()

        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f + %.5f], recall=[%.5f, %.5f], ' \
                       'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                       (epoch, t2 - t1, t3 - t2, loss, base_loss, DIN_loss, reg_loss, ret['recall'][0], ret['recall'][-1],
                        ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                        ret['ndcg'][0], ret['ndcg'][-1])
            print(perf_str)


    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)

    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                 (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in pres[idx]]),
                  '\t'.join(['%.5f' % r for r in hit[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    print(final_perf)

    save_path = '%soutput/%s/%s.result' % (args.proj_path, args.dataset, model.model_type)
    ensureDir(save_path)
    f = open(save_path, 'a')

    f.write('embed_size=%d, lr=%.4f, layer_size=%s, node_dropout=%s, mess_dropout=%s, regs=%s, loss_type=%s,'
            'adj_type=%s, use_att=%s, use_DIN=%s\n\t%s\n'
            % (args.embed_size, args.lr, args.layer_size, args.node_dropout, args.mess_dropout, args.regs, args.model_type,
            args.adj_type, args.use_att, args.use_DIN, final_perf))
    f.close()
    drug_embed, entity_embed, relation_embed = sess.run(
        [model.weights['drug_embed'], model.weights['entity_embed'], model.weights['relation_embed']],
        feed_dict={})

    temp_save_path = '%spretrain/%s/%s.npz' % (args.proj_path, args.dataset, args.model_type)
    ensureDir(temp_save_path)
    np.savez(temp_save_path, drug_embed=drug_embed, entity_embed=entity_embed, relation_embed=relation_embed)
    print('save the embed of model in path: ', temp_save_path)
