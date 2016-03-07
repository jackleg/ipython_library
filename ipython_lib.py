#!/usr/bin/env python
#!coding=utf8

import sys, urllib, urllib2, os, subprocess, time, math, random
import re
import threading
import json

try:
    import matplotlib.pyplot as plt
except ImportError: 
    plt = None

import pandas as pd
from pandas import DataFrame, Series
import numpy as np
from IPython.display import *
import functools
import copy
import warnings
import matplotlib.pyplot as plt


def aa():
    return np.random.random()

pd.set_option('display.width', 2000)
pd.set_option('max_colwidth', 800)
pd.set_option('max_rows', 200)

from collections import *
##FeatureStrings = namedtuple('FeatureStrings', ['bm25', 'prox'])

sys.path.append('../lib')
try: import api_tools; reload(api_tools);
except: pass

try: import image_tools; reload(image_tools);
except: pass

try: import common_tools; reload(common_tools);
except: pass


def to_img_tag(img_url):
    return '<a hred=%s><img src=%s></a>' % (img_url, img_url)


def page(df, n, p):
    if len(df) > 0:
        s = n*p
        e = min(n*(p+1), len(df))
        return df.iloc[s:e]
    else:
        return df


def shuffle(df, reset_index=False, seed=None):
    if reset_index:
        df = df.reset_index(drop=True)
    np.random.seed(seed)
    return df.reindex(np.random.permutation(df.index))


def divide(X, r, div_label=[0, 1], seed=None, div_name='div'):
    """X를 랜덤하게 두개의 div로 나눈다. 

    seed를 주면 랜덤하되, 항상 동일한 그룹으로 나누어준다. (동일한 입력에 대해서)
    """
    X[div_name] = div_label[1]
    if seed is not None:
        np.random.seed(seed)
    X = shuffle(X)
    X.iloc[:int(len(X)*r)][div_name] = div_label[0]

    return X


def sample_df(df, k, replace=True):
    if replace:
        return df.iloc[np.random.randint(0, len(df), size=k)]


def ith_group(df, fn, ith, max_n=20, do_shuffle=False):
    A = df[df[fn] == df[fn].unique()[ith]]
    if do_shuffle:
        A = shuffle(A)

    if max_n != None and max_n < len(A):
        A = A.iloc[:max_n]

    return A


def sample_with_count(counts, sample_rate, seed=None):
    if seed is not None:
        np.random.seed(seed)
    r = np.random.random(len(counts))
    selected = r < (1.0 - (1.0 - sample_rate)**counts)
    return selected


###############################################################################
# plotting
###############################################################################

def get_log_odds_bak(target, feature, bins, f_range=None, M=0.5):
    '''return log10 ( P(feature=x | target=1) / P(feature=x | target=0) )
       tn : targent name, 0 or 1
       fn : x name 
       f_range : x의 범위 제한
       M : smoothing factor
    '''
    tn = target.name
    fn = feature.name
    X = pd.concat([target, feature], axis=1)
    if f_range is not None:
        X = X[(X[fn]>f_range[0]) & (X[fn]<f_range[1])]
    X['_cut'] = pd.cut(X[fn], bins=bins)
    X['_cut'] = X._cut.map(lambda x: float(x.split(',')[0][1:]))
    Y = X.groupby('_cut').apply(lambda x: np.log10((x[tn].sum() + M) / ((1.0-x[tn]).sum() + M)))
    Y = Y - np.log10(1.0 * X[tn].sum() / (1.0-X[tn]).sum())
    Y = pd.DataFrame(Y, columns=['%s_log_odds' % fn])
    return Y


def get_log_odds(target, feature, bins, f_range=None, M=10, display_head=False):
    '''return log ( P(feature=x | target=1) / P(feature=x | target=0) )
       tn : targent name, 0 or 1
       fn : x name 
       f_range : x의 범위 제한
       M : smoothing factor
    '''
    tn = target.name
    fn = feature.name
    X = pd.concat([target, feature], axis=1)
    if f_range is not None:
        X = X[(X[fn]>f_range[0]) & (X[fn]<f_range[1])]
    if display_head:
        X['_cut'] = pd.cut(X[fn], bins=bins).astype(str)
        X['_cut'] = X._cut.map(lambda x: float(x.split(',')[0][1:]))
    else:
        X['_cut'] = pd.cut(X[fn], bins=bins)
    Y = X.groupby('_cut').apply(lambda x: np.log((x[tn].sum() + 1.0*M/bins) / ((1.0-x[tn]).sum() + 1.0*M/bins)))
#    display(X.groupby('_cut').apply(lambda x: (x[tn].sum(), (1-x[tn]).sum())))
#    display(Y)
    Y = Y - np.log( (1.0 * X[tn].sum() + M) / ( (1.0-X[tn]).sum() + M) )
    Y = pd.DataFrame(Y, columns=['%s_log_odds' % fn])
    return Y


def plot_log_odds(target, feature, bins, f_range=None, M=10, figsize=(10, 3), display_head=False, normed=True):
    LO = get_log_odds(target, feature, bins, f_range=f_range, M=M, display_head=display_head)
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    ax = axs[0]; feature[target==1].hist(bins=bins, alpha=0.4, color='red', normed=normed, range=f_range, ax=ax)
    ax = axs[0]; feature[target==0].hist(bins=bins, alpha=0.4, color='blue', normed=normed, range=f_range, ax=ax)
    ax = axs[1]; LO.plot(ax=ax)


def heatmap(df, cmap="OrRd", figsize=(10, 10)):
    """draw heatmap of df"""

    plt.figure(figsize=figsize)
    plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns)
    plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
    plt.pcolor(df, cmap=cmap)

###############################################################################
# data frame display
###############################################################################

def hist_df(df, columns=None, quantiles=None, **kwargs):
    kwargs['bins'] = kwargs.get('bins', 20)
    kwargs['alpha'] = kwargs.get('alpha', 0.5)

    if columns is None:
        columns = df.columns

    if not isinstance(quantiles, list):
        quantiles = [quantiles] * len(columns)

    figsize = kwargs.get('figsize', (2+4*len(columns), 4))

    fig, axs = plt.subplots(1, len(columns), figsize=figsize)
    for i in range(len(columns)):
        col = columns[i]
        q   = quantiles[i]
        if len(columns) == 1:
            ax = axs
        else: 
            ax  = axs[i]; 
        if q is not None:
            q_range = (df[col].quantile(q), df[col].quantile(1.0 - q))
        else:
            q_range = None
        df[col].hist(ax=ax, range=q_range, **kwargs)
        ax.set_title(col)
        


###############################################################################
# data frame save/load
###############################################################################

def save_df(df):
    pass


def load_df(df_path):
    pass

###############################################################################
# data frame display
###############################################################################

def _get_df_part(df, **kwargs):
    """data frame의 부분을 돌려줌

    shuffled : shuffle할지 여부 (default : False)
    p : page (default : 0) 
    n : number of row in page (default : None(all))
    fields : columns (default : None(all))
    """
    if len(df) == 0: return df

    # shuffle
    shuffled = kwargs.get('shuffled', False)
    if shuffled:
        df = shuffle(df)

    # paging
    page = kwargs.get('p', 0)
    page_size = kwargs.get('n', None)
    if page_size is not None:
        start = min(page    *page_size, len(df)-1)
        end   = min((page+1)*page_size, len(df))
        df = df.iloc[start:end]
        
    # fiels
    fields = kwargs.get('fields', None)
    if fields is not None:
        df = df[fields]

    return df


def add_script(name, func_name, html, df_name='None'):
    d = {}
    d['name'] = name
    d['func_name'] = func_name
    d['html'] = html
    d['df_name'] = df_name
    new_html = '''
        <div id="%(name)s">
        %(html)s
        </div>
        
        <script>
        $("#%(name)s").buttonset();

        $('#%(name)s :radio').click(function() {
            var kernel = IPython.notebook.kernel;
            kernel.execute("%(func_name)s('" + $(this).attr('name') + "', '" + $(this).attr('id') + "', %(df_name)s)" );
            ;
        });
        </script>
    ''' % d
    return new_html


def make_radio_button(name, r_ids):
    ''' <input type='radio' name=name, id=r_id> 
    r_ids = [r_id, ...]    
    '''
    s_list = map(lambda r_id: """<input type="radio" id="%s" name="%s"> %s<br>""" % (r_id, name, r_id), r_ids)
    tag = ' '.join(s_list)
    return tag

def get_df_with_radio_button(A, key_column, values):
    A[key_column+'_eval'] = A[key_column].map(lambda key: make_radio_button(key, values))
    return A


def complete_img_df(df, txt_fields=None, img_fields=['thumbnail'], link_fields=[], 
                   use_api=True, img_first=True, docid_name='dsid', **kwargs):
    return _complete_img_df(df, txt_fields, img_fields, link_fields, use_api, img_first, docid_name, **kwargs)


## image를 포함한 data frame을 html으로 변환해서 출력
def _complete_img_df(df, txt_fields=None, img_fields=['thumbnail'], link_fields=[], 
                   use_api=True, img_first=True, docid_name='dsid', **kwargs):
##    print 'data_frame size : %s' % len(df)
    if len(df) == 0: return 
    df = df.copy()

    if txt_fields is None:
        txt_fields = list(df.columns)
        for img_field in img_fields:
            try:
                txt_fields.remove(img_field)
            except:
                pass
        for link_field in link_fields:
            try:
                txt_fields.remove(link_field)
            except:
                pass
    abs_fields = list((set(txt_fields).union(set(img_fields)).union(set(link_fields))) - 
                       set(df.columns))

    # 모든 데이터를 사용하진 않으니, 필요한 데이터를 먼저 추려낸다. 
    # image 기본 보여주는 갯수가 많으면 부하가 클 수 있으므로
    # 기본 값은 안전하게 세팅한다. 
    if kwargs.get('n', None) is None:
        kwargs['n'] = 10
    df = _get_df_part(df, **kwargs)

##    print len(df)
    if len(abs_fields) > 0 and use_api:
        dsids = list(df[docid_name])
        abs_info = api_tools.get_img_info_by_docids_df(dsids, abs_fields)
        if len(abs_info) == 0:
            for abs_field in abs_fields:
                ## 아르곤에서 받아온 데이터가 한건도 없을 경우 join이 되지 않으므로 임시 처리
                df[abs_field] = ''         
        else:
            df = pd.merge(df, abs_info, left_on=docid_name, right_index=True, how='left')
    
    if img_first:
        dsp_fields = img_fields + txt_fields
    else:
        dsp_fields = txt_fields + img_fields 

    width = kwargs.get('width', 200)
    height = kwargs.get('height', None)
    for img_field in img_fields:
        df[img_field] = df[img_field].map(lambda x: to_img_src(x, width, height))

    for link_field in link_fields:
        df[link_field] = df[link_field].map(lambda x: '<a href="%s" target="new">link</a>' % (x, ))

    return df[dsp_fields]
    
    
def to_img_src(img_url, width=None, height=None):
    if width is not None:
    	width_str = ' width="%s"' % width
    else:
    	width_str = ''
    if height is not None:
    	height_str = ' height="%s"' % height
    else:
    	height_str = ''
        
    img_src = "<a href='%s' target=new2><img src='%s'%s%s></a>" % (img_url, img_url, width_str, height_str)
    return img_src


def display_df(df, html=False, **kwargs):
    print 'total : %s' % len(df)
    
    df = _get_df_part(df, **kwargs)

    # html (충분히 길고 넓은 데이터를 보여줄 수 있음, image 표현 가능)
    if html:
        display(HTML(df.to_html()))
    else:
        display(df)


def get_img(dsid=None):
    if dsid is not None:
        thumnail_url = api_tools.get_img_info_by_docid(dsid, 'thumbnail')
        return Image(url=thumnail_url)


def display_img_df(df, txt_fields=None, img_fields=['thumbnail'], link_fields=[],  
                   use_api=True, img_first=True, docid_name='dsid', 
                   script_func_name=None, script_name=None, **kwargs):
    ''' image display를 위해서 html로 변환하여 화면에 표시

    img_fields에 정의된 필드는 excaping하지 않는다. 
    use_api : dsid(docimgdsid) 필드가 있으면 가 있으면 
              api로 필요한 데이터를 가져온다. 
    '''

    if len(df) == 0:
        return 

    df = _complete_img_df(df, txt_fields, img_fields, link_fields, use_api, img_first, docid_name=docid_name, **kwargs)
    html = df.to_html(escape=False)
    df_name = kwargs.get('script_df_name', 'None')
    if script_func_name is not None:
        if script_name is None:
            script_name = script_func_name
    	html = add_script(script_func_name, script_func_name, html, df_name=df_name)
    display(HTML(html))


try:
    import test_anal; reload(test_anal)
except:
    test_anal = None
def display_table_info(table_name, job_name=None, args_str='', n=10, local=False, 
                       use_all=True, html_n=0, **kwargs):
    A = test_anal.head_table(table_name, job_name=job_name, args_str=args_str, n=n, 
                                local=local, use_all=use_all, **kwargs)
    display(A)

    if html_n > 0:
        display(HTML(A.head(html_n).to_html()))

    return A


def get_table_df(table_name, job_name=None, n=10, local=False, use_all=True, **kwargs):
    A = test_anal.head_table(table_name, job_name=job_name, n=n, 
                                local=local, use_all=use_all, **kwargs)
    return A

    
def keep_alive():
    from IPython.core.display import HTML
    HTML(get_keep_alive_str())


def get_keep_alive_str(timeout=100000):
    """ Return script string for keep alive.

    ipython에서 rendering하면서 keep alive script가 실행되는 것을 이용. 
    ex) HTML(ipython_lib.get_keep_alive(3600)) 과 같이 사용
    주의 : 정확한 원인은 모르지만, 저장 후 리로드를 하,면 
    위 문장이 실행된 아랫부분의 코드는 없어짐 
    (원인을 찾기 전까지 notebook 가장 아랫쪽에 사용 권유)
    """
    keep_alive_str = '''<script>
        var keepalive_timer;
        function keep_connection_alive() {
        IPython.notebook.kernel.execute();
        keepalive_timer = setTimeout('keep_connection_alive()', %s);
        }
        keep_connection_alive();
        </script>''' % timeout

    return keep_alive_str


def no_warning():
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="pandas", lineno=570)


def df_from_hdfs(path, columns=None, n=10):
    if n is None:
        cmd = 'hadoop fs -text %s' % (path, )
    else:
        cmd = 'hadoop fs -text %s | head -n %s' % (path, n)

    pipe = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).stdout

    return pd.read_csv(pipe, sep="\t", names=columns)


def to_html(A, link_fields=[], **kwargs):
    dsp_url_max_len = kwargs.get("dsp_url_max_len", 40)
    def _make_short_url(url):
        if isinstance(url, float):
            return url
        if len(url) <= dsp_url_max_len:
            return url
        else:
            return url[:dsp_url_max_len-3] + '...'

    if not isinstance(link_fields, list) and not isinstance(link_fields, tuple):
        link_fields = [link_fields]
    B = A.copy()
    for link_field in link_fields:
        B[link_field] = B[link_field].map(lambda x: '<a href="%s" target="new">%s</a>' % (x, _make_short_url(x)))
    return HTML(B.to_html(escape=False))


def flatten_for_column(A, vfn, index_name):
    '''만들었으나, 잘 사용하지 않음
        use flatten_column
    '''
    B = A.groupby(level=0, as_index=False, group_keys=False).apply(lambda x: DataFrame({index_name:x.index[0], vfn+"_flt":list(x[vfn].iloc[0])}))
    return pd.merge(A, B, left_index=True, right_on=index_name)


def flatten_column(A, fn, new_fn, remain_fields=None):
    Y = pd.DataFrame([[i, x] 
               for i, y in A[fn].apply(list).iteritems() 
                    for x in y], columns=['I', new_fn])
    Y = Y.set_index('I')
    if remain_fields is None:
        return A.join(Y)    
    else:
    	return A[remain_fields].join(Y)


try:
    import search_lib
except:
    pass
def crawl_and_save_search_result(svc, query_list, save_path, doc_fields, meta_fields=[], n=10, opt=None, verbose=False, interval=100, break_cnt=None):
    save_dir = save_path.rsplit('/', 1)[0]
    common_tools.check_dir(save_dir)

    R_list = []
    if verbose:
        timer = common_tools.Timer(interval, break_cnt)
    for query in query_list:
        try:
            if verbose and timer.check():
                break
            if svc.startswith('image'):
                svc_type = svc.split('_', 1)[1]
                meta, ds = search_lib.get_image_search_result(query, svc_type=svc_type, n=n, opt=opt)
                # FIXME 검색결과 없음은 일단 pass, 추가로 결과없음 정보를 어떻게 남길지는 고민할 것
                if len(ds) == 0:
                    continue
                R = pd.DataFrame(ds)[doc_fields]
                for meta_field in meta_fields:
                    R[meta_field] = meta[meta_field]
                
            R_list.append(R)
        except Exception, ex:
            sys.stderr.write('%s\n' % ex)

    T = pd.concat(R_list).reset_index(drop=True)
    T.to_pickle(save_path)
    if verbose:
        sys.stderr.write('Jobs done\n')


from sklearn import metrics
def plot_precision_recall_curves(target, feature, ax, sample_weight=None, color='blue', fn=''):
    pr, rc, thresholds = metrics.precision_recall_curve(target, feature, pos_label=1, sample_weight=sample_weight)
    average_precision_score = metrics.average_precision_score(target, feature, sample_weight=sample_weight)
    ax.plot(rc, pr, label='%s : %.3f' % (fn, average_precision_score), color=color)
    ax.set_xlabel('Recal')
    ax.set_ylabel('Precision')
    ax.legend(loc='best')


def plot_precision_recall_curves_with_thres(target, feature, ax, sample_weight=None, fn=''):
    pr, rc, thresholds = metrics.precision_recall_curve(target, feature, pos_label=1, sample_weight=sample_weight)
    ax.plot(thresholds, pr[:-1], label='precision', color='red')
    ax.plot(thresholds, rc[:-1], label='recall', color='blue')
    ax.set_xlabel('Thresholds')
    ax.set_ylabel('Precision/Recall')
    ax.legend(loc='best')


def plot_roc_curve(target, feature, ax, sample_weight=None, color='blue', fn=''):
    fpr, tpr, thresholds = metrics.roc_curve(target, feature, pos_label=1, sample_weight=sample_weight)
    ax.plot(fpr, tpr, label='%s' % fn, color=color)
    ax.set_xlabel('fpr')
    ax.set_ylabel('tpr')
    ax.legend(loc='best')


def plot_roc_curve_with_thres(target, feature, ax, sample_weight=None, fn=''):
    fpr, tpr, thresholds = metrics.roc_curve(target, feature, pos_label=1, sample_weight=sample_weight)
    ax.plot(thresholds, fpr, label='fpr', color='red')
    ax.plot(thresholds, tpr, label='tpr', color='blue')
    ax.set_xlabel('thresholds')
    ax.set_ylabel('tpr/fpr')
    ax.legend(loc='best')


def from_unicode_to_str(x, encoding='utf8'):
    if isinstance(x, unicode):
        return x.encode(encoding)
    else:
        return x


def from_all_to_str(x, encoding='utf8'):
    if isinstance(x, unicode):
        return x.encode(encoding)
    else:
        return str(x)


def calc_ndid64_dist(a, b):
    return bin(a ^ b).count('1')


if __name__ == "__main__":
    no_warning()
    if len(sys.argv) > 1: 
        func_name = sys.argv[1].strip()
        arg_list = map(lambda x: "'%s'" % x.strip(), sys.argv[1:])
        exec_str = "%s(%s)" % (func_name, ",".join(arg_list))
        exec(exec_str)


###############################################################################
# display
###############################################################################
def display_full(df):
    """
    truncation view로 들어가지 않고, df를 모두 출력한다.
    """
    with pd.option_context("display.max_rows", len(df)), pd.option_context("display.max_columns", len(df.columns)):
        display(df)

###############################################################################
# data_util
###############################################################################
def read_tsv(filepath_or_buffer, **kwargs):
    """seperator가 tab인 csv 읽기"""

    return pd.read_csv(filepath_or_buffer, sep="\t", **kwargs)

