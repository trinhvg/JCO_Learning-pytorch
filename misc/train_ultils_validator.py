import io
import itertools
import json
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import random
import re
import shutil
import textwrap

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from termcolor import colored

import torch
import torch.nn as nn
import torch.nn.functional as F
import imgaug as ia
from scipy.special import softmax
from sklearn.metrics import classification_report

def check_manual_seed(seed):
    """
    If manual seed is not specified, choose a random one and notify it to the user
    """
    seed = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    ia.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    print('Using manual seed: {seed}'.format(seed=seed))
    return


def check_log_dir(log_dir):
    # check if log dir exist
    if os.path.isdir(log_dir):
        color_word = colored('WARMING', color='red', attrs=['bold', 'blink'])
        print('%s: %s exist!' % (color_word, colored(log_dir, attrs=['underline'])))
        while (True):
            print('Select Action: d (delete)/ q (quit)', end='')
            key = input()
            if key == 'd':
                shutil.rmtree(log_dir)
                break
            elif key == 'q':
                exit()
            else:
                color_word = colored('ERR', color='red')
                print('---[%s] Unrecognized character!' % color_word)
    return


def plot_confusion_matrix(conf_mat, label):
    """
    Parameters:
        title='Confusion matrix'        : Title for your matrix
        tensor_name = 'MyFigure/image'  : Name for the output summay tensor
     Returns:
        summary: image of plot figure
    Other items to note:
        - Depending on the number of category and the data , you may have to modify the figzie, font sizes etc.
        - Currently, some of the ticks dont line up due to rotations.
    """

    cm = conf_mat

    np.set_printoptions(precision=2)  # print numpy array with 2 decimal places

    fig = matplotlib.figure.Figure(figsize=(7, 7), dpi=320, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, cmap='Oranges')

    classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in label]
    classes = ['\n'.join(textwrap.wrap(l, 40)) for l in classes]

    tick_marks = np.arange(len(classes))

    ax.set_xlabel('Predicted', fontsize=7)
    ax.set_xticks(tick_marks)
    c = ax.set_xticklabels(classes, fontsize=4, rotation=-90, ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label', fontsize=7)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=4, va='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd') if cm[i, j] != 0 else '.',
                horizontalalignment="center", fontsize=6,
                verticalalignment='center', color="black")
    fig.set_tight_layout(True)

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()

    # get PNG data from the figure
    png_buffer = io.BytesIO()
    fig.canvas.print_png(png_buffer)
    png_encoded = png_buffer.getvalue()
    png_buffer.close()

    return png_encoded


####
def update_log(output, epoch, net_name, prefix, color, tfwriter, log_file, logging):
    # print values and convert
    max_length = len(max(output.keys(), key=len))
    for metric in output:
        key = colored(prefix + '-' + metric.ljust(max_length), color)
        print('------%s : ' % key, end='')
        if metric not in ['conf_mat_c', 'conf_mat_r', 'box_plot_data']:
            print('%0.7f' % output[metric])
        elif metric == 'conf_mat_c':
            conf_mat_c = output['conf_mat_c']  # use pivot to turn back
            conf_mat_c_df = pd.DataFrame(conf_mat_c)
            conf_mat_c_df.index.name = 'True'
            conf_mat_c_df.columns.name = 'Pred'
            output['conf_mat_c'] = conf_mat_c_df
            print('\n', conf_mat_c_df)
        elif metric == 'conf_mat_r':
            conf_mat_r = output['conf_mat_r']  # use pivot to turn back
            conf_mat_r_df = pd.DataFrame(conf_mat_r)
            conf_mat_r_df.index.name = 'True'
            conf_mat_r_df.columns.name = 'Pred'
            output['conf_mat_r'] = conf_mat_r_df
            print('\n', conf_mat_r_df)
        elif metric == 'box_plot_data':
            box_plot_data = output['box_plot_data']  # use pivot to turn back
            box_plot_data_df = pd.DataFrame(box_plot_data)
            box_plot_data_df.columns.name = 'Pred'
            output['box_plot_data'] = box_plot_data_df

    if not logging:
        return

    # create stat dicts
    stat_dict = {}
    for metric in output:
        if metric not in ['conf_mat_c', 'conf_mat_r', 'box_plot_data']:
            metric_value = output[metric]
        elif metric == 'conf_mat_c':
            conf_mat_df = output['conf_mat_c']  # use pivot to turn back
            conf_mat_df = conf_mat_df.unstack().rename('value').reset_index()
            conf_mat_df = pd.Series({'conf_mat_c': conf_mat_c}).to_json(orient='records')
            metric_value = conf_mat_df
        elif metric == 'conf_mat_r':
            conf_mat_regres_df = output['conf_mat_r']  # use pivot to turn back
            conf_mat_regres_df = conf_mat_regres_df.unstack().rename('value').reset_index()
            conf_mat_regres_df = pd.Series({'conf_mat_r': conf_mat_r}).to_json(orient='records')
            metric_value = conf_mat_regres_df
        elif metric == 'box_plot_data':
            box_plot_data_df = pd.Series({'box_plot_data': box_plot_data}).to_json(orient='records')
            metric_value = box_plot_data_df
        stat_dict['%s-%s' % (prefix, metric)] = metric_value

    # json stat log file, update and overwrite
    with open(log_file) as json_file:
        json_data = json.load(json_file)

    current_epoch = str(epoch)
    current_model = str(net_name)
    if current_epoch in json_data:
        old_stat_dict = json_data[current_model]
        stat_dict.update(old_stat_dict)
    current_epoch_dict = {current_model: stat_dict}
    json_data.update(current_epoch_dict)

    with open(log_file, 'w') as json_file:
        json.dump(json_data, json_file)

    # log values to tensorboard
    for metric in output:
        if metric not in ['conf_mat_c', 'conf_mat_r', 'box_plot_data']:
            tfwriter.add_scalar(prefix + '-' + metric, output[metric], current_epoch)


####
def log_train_ema_results(engine, info):
    """
    running training measurement
    """
    training_ema_output = engine.state.metrics  #
    training_ema_output['lr'] = float(info['optimizer'].param_groups[0]['lr'])
    update_log(training_ema_output, engine.state.epoch, 'train-ema', 'green',
               info['tfwriter'], info['json_file'], info['logging'])


####
def process_accumulated_output_multi(output, batch_size, nr_classes):
    #
    def uneven_seq_to_np(seq):
        item_count = batch_size * (len(seq) - 1) + len(seq[-1])
        cat_array = np.zeros((item_count,) + seq[0][0].shape, seq[0].dtype)
        # BUG: odd len even
        if len(seq) < 2:
            return seq[0]
        for idx in range(0, len(seq) - 1):
            cat_array[idx * batch_size:
                      (idx + 1) * batch_size] = seq[idx]
        cat_array[(idx + 1) * batch_size:] = seq[-1]
        return cat_array

    proc_output = dict()
    true = uneven_seq_to_np(output['true'])
    # threshold then get accuracy
    if 'logit_c' in output.keys():
        logit_c = uneven_seq_to_np(output['logit_c'])
        pred_c = np.argmax(logit_c, axis=-1)
        # pred_c = [covert_dict[pred_c[idx]] for idx in range(len(pred_c))]
        acc_c = np.mean(pred_c == true)
        print(acc_c)
        # confusion matrix
        conf_mat_c = confusion_matrix(true, pred_c, labels=np.arange(nr_classes))
        proc_output.update(acc_c=acc_c, conf_mat_c=conf_mat_c,)
    if 'logit_r' in output.keys():
        logit_r = uneven_seq_to_np(output['logit_r'])
        label = np.transpose(np.array([[0., 1., 2., 3.]]).repeat(len(true), axis=0), (1, 0))
        pred_r = np.argmin(abs((logit_r - label)), axis=0)
        # pred_r = [covert_dict[pred_r[idx]] for idx in range(len(pred_r))]
        acc_r = np.mean(pred_r == true)
        # print(acc_r)
        # confusion matrix
        conf_mat_r = confusion_matrix(true, pred_r, labels=np.arange(nr_classes))
        proc_output.update(acc_r=acc_r, conf_mat_r=conf_mat_r)

    # proc_output.update(box_plot_data=np.concatenate(
    #         [true[np.newaxis, :], pred_c[np.newaxis, :], pred_r[np.newaxis, :], logit_r.transpose(1, 0)], 0))
    return proc_output

def process_accumulated_output_multi_mix(output, batch_size, nr_classes):
    #
    def uneven_seq_to_np(seq):
        item_count = batch_size * (len(seq) - 1) + len(seq[-1])
        cat_array = np.zeros((item_count,) + seq[0][0].shape, seq[0].dtype)
        # BUG: odd len even
        if len(seq) < 2:
            return seq[0]
        for idx in range(0, len(seq) - 1):
            cat_array[idx * batch_size:
                      (idx + 1) * batch_size] = seq[idx]
        cat_array[(idx + 1) * batch_size:] = seq[-1]
        return cat_array

    proc_output = dict()
    true = uneven_seq_to_np(output['true'])
    # threshold then get accuracy
    if 'logit_c' in output.keys():
        logit_c = uneven_seq_to_np(output['logit_c'])

        pred_c = np.argmax(logit_c, axis=-1)
        # pred_c = [covert_dict[pred_c[idx]] for idx in range(len(pred_c))]
        acc_c = np.mean(pred_c == true)
        print('acc_c',acc_c)
        # print(classification_report(true, pred_c, labels=[0, 1, 2, 3]))
        # confusion matrix
        conf_mat_c = confusion_matrix(true, pred_c, labels=np.arange(nr_classes))
        proc_output.update(acc_c=acc_c, conf_mat_c=conf_mat_c,)
    if 'logit_r' in output.keys():
        logit_r = uneven_seq_to_np(output['logit_r'])
        label = np.transpose(np.array([[0., 1., 2., 3.]]).repeat(len(true), axis=0), (1, 0))
        pred_r = np.argmin(abs((logit_r - label)), axis=0)
        # pred_r = [covert_dict[pred_r[idx]] for idx in range(len(pred_r))]
        acc_r = np.mean(pred_r == true)
        print('acc_r',acc_r)
        # print(classification_report(true, pred_r, labels=[0, 1, 2, 3]))
        # confusion matrix
        conf_mat_r = confusion_matrix(true, pred_r, labels=np.arange(nr_classes))
        proc_output.update(acc_r=acc_r, conf_mat_r=conf_mat_r)

    # if ('logit_r' in output.keys()) and ('logit_c' in output.keys()):
    #     a = abs((logit_r - label)).transpose(1, 0)
    #     prob_r = softmax(-a, 1)
    #     logit_c +=prob_r
    #     pred_c = np.argmax(logit_c, axis=-1)
    #     acc_c = np.mean(pred_c == true)
    #     print('acc_mix',acc_c)

    # proc_output.update(box_plot_data=np.concatenate(
    #         [true[np.newaxis, :], pred_c[np.newaxis, :], pred_r[np.newaxis, :], logit_r.transpose(1, 0)], 0))
    return proc_output

def process_accumulated_output_multi_testAUG(output, batch_size, nr_classes):
    #
    def uneven_seq_to_np(seq):
        item_count = batch_size * (len(seq) - 1) + len(seq[-1])
        cat_array = np.zeros((item_count,) + seq[0][0].shape, seq[0].dtype)
        # BUG: odd len even
        for idx in range(0, len(seq) - 1):
            cat_array[idx * batch_size:
                      (idx + 1) * batch_size] = seq[idx]
        cat_array[(idx + 1) * batch_size:] = seq[-1]
        return cat_array

    proc_output = dict()
    true = uneven_seq_to_np(output['true'])
    # threshold then get accuracy
    if 'pred_c' in output.keys():
        pred_c = uneven_seq_to_np(output['pred_c'])
        acc_c = np.mean(pred_c == true)
        # confusion matrix
        conf_mat_c = confusion_matrix(true, pred_c, labels=np.arange(nr_classes))
        proc_output.update(acc_c=acc_c, conf_mat_c=conf_mat_c,)
    if 'pred_r' in output.keys():
        pred_r = uneven_seq_to_np(output['pred_r'])
        acc_r = np.mean(pred_r == true)
        # confusion matrix
        conf_mat_r = confusion_matrix(true, pred_r, labels=np.arange(nr_classes))
        proc_output.update(acc_r=acc_r, conf_mat_r=conf_mat_r)
    return proc_output


####
def inference(engine, inferer, prefix, dataloader, info):
    """
    inference measurement
    """
    inferer.accumulator = {metric: [] for metric in info['metric_names']}
    inferer.run(dataloader)
    output_stat = process_accumulated_output_multi(inferer.accumulator,
                                             info['infer_batch_size'], info['nr_classes'])
    update_log(output_stat, engine.state.epoch, prefix, 'red',
               info['tfwriter'], info['json_file'], info['logging'])
    return


####
def accumulate_outputs(engine):
    batch_output = engine.state.output
    for key, item in batch_output.items():
        engine.accumulator[key].extend([item])
    return


def accumulate_predict(pred_patch):
    unique, counts = np.unique(pred_patch.cpu(), return_counts=True)
    pred_count = dict(zip(unique, counts))
    patch_label = max(pred_count, key=pred_count.get)
    return patch_label
