""" The code in 'utils.py' is borrowed from the code for DeepHit model. The github link of the code for DeepHit is https://github.com/chl8856/DeepHit. Reference: C. Lee, W. R. Zame, J. Yoon, M. van der Schaar, "DeepHit: A Deep Learning Approach to Survival Analysis with Competing Risks," AAAI Conference on Artificial Intelligence (AAAI), 2018.

"""


import logging
import os
import fnmatch
import re
from subprocess import call
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics import roc_auc_score, average_precision_score


def init_logger(odir='.', log_fn='log.txt', use_show=True, log_level=None):
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logging.basicConfig(level=logging.WARNING, format='%(message)s')
    log_fn = "{}/{}".format(odir , log_fn)
    #if os.path.isfile(log_fn):
        #os.remove(log_fn)
    handler = logging.FileHandler(log_fn)
    if log_level is None:
        log_level = logging.INFO if use_show else logging.WARNING
    handler.setLevel(log_level)
    logger.addHandler(handler)
    logger.setLevel(log_level)
    return logger


def get_proj_dir():
    return os.path.abspath(os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        os.pardir))


def get_data_dir():
    return os.path.join(get_proj_dir(), 'data')


def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)


def find_file_dir(dir, fn_mask):
    file_lst = []
    for dir_name, sub_dir, f_list in os.walk(dir):
        for file_name in f_list:
            if fnmatch.fnmatch(file_name, fn_mask):
                file_lst.append(os.path.join(dir_name, file_name))
    return file_lst


def col_with_nan(df):
    col_nan = []
    for el in df.columns:
        if sum(df[el].isnull()):
            col_nan.append(el)
    return col_nan


def exe_cmd(logger, cmd, assert_on_error=True):
    logger.info('cmd:{}'.format(cmd))
    cmd = re.sub(' +', ' ', cmd).rstrip().lstrip()
    cmd_lst = cmd.split(' ')
    rval = call(cmd_lst)
    if assert_on_error:
        assert rval == 0


def count_properties(a):
    d = Counter(a)
    rval_d = dict()
    sum_val = sum(d.values())
    for el in d.keys():
        rval_d['{}_ratio'.format(el)] = d[el]/float(sum_val)
    kys = d.keys()
    for el in kys:
        rval_d[el] = d[el]
    return rval_d



def log_meminfo():
    fn = '/proc/meminfo'
    if os.path.isfile(fn):
        f = open(fn, "r")
        for ln in f:   
            logger.info('{}'.format(ln))
        f.close()


def get_hostname():
    return os.environ['HOSTNAME'] if 'HOSTNAME' in os.environ else 'unknown'


def get_df_compression(fn):
    return 'gzip' if fn.endswith('.gz') else None


logger = logging.getLogger()


