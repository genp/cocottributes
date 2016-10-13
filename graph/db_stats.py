import json
import random
from collections import Counter

import matplotlib.pyplot as plt
from sklearn.externals import joblib

from app import app, db
from app.models import * 
from utils import utils
from mturk import filter_workers


def get_ap_scores(train_percent, feat_type, sname):
    ap = {}
    cat_ids = [-1, 1, 91, 93, 97]
    for cat_id in cat_ids:
        
        stmt = "select distinct label_id, name, cat_id, ap, num_pos::float/(num_pos+num_neg) as chance, num_pos, type, location from classifier_score c, label lbl where c.label_id = lbl.id and train_percent = {0} and cat_id = {1} and type = '{2}'".format(train_percent, cat_id, feat_type)
        ap[cat_id] = db.engine.execute(stmt).fetchall()

    joblib.dump(ap, sname)


def label_popularity(label_ids, patch_ids):
    # val = Annotation.query.filter(Annotation.patch_id == cmd['patches'][0]['id']).filter(Annotation.label_id == cmd['attributes'][0]['id']).all()
    # [ x.value for x in val]
    num_votes_per_label = {}
    trend_votes_per_label = {}
    for lid in label_ids:
        print 'Label %s...' % Label.query.get(lid).name
        num_votes = 0

        trend_votes_per_label[lid] = []

        for pid in patch_ids:
            vals = [x[0] for x in utils.get_all_db_res('select value from annotation a, hit_response h, worker w '\
                                                       'where a.hit_id = h.id and a.patch_id = %d and a.label_id = %d '\
                                                       'and h.worker_id = w.id and w.is_blocked is false' % (pid, lid))]
            if len(vals) < 3:
                continue
            c = Counter(vals)
            tally = c.items()
            vote = max(tally, key=lambda x: x[1])
            trend_votes_per_label[lid].append(float(vote[1])/len(vals))
            if vote[0]:
                num_votes += 1

        num_votes_per_label[lid] = num_votes

    return num_votes_per_label, trend_votes_per_label
                
def compare_sun_attrs(label_ids, patch_ids):
    sun_attrs = joblib.load('data/sun_attrs.jbl')
    sun_labels = sun_attrs['labels_cv']
    # TODO makes some plots
    num_votes_per_label, trend_votes_per_label = label_popularity(label_ids, patch_ids)

    pop_sun = []
    pop_coco = []
    for a_idx, attr in enumerate(sun_attrs['attributes']):
        lid = utils.get_first_db_res("select id from label where parent_id = 102 and name like '"+attr+"%%'")
        pop_sun.append(sum(sun_labels[:,a_idx])/len(sun_labels[:,a_idx]))
        pop_coco.append(num_votes_per_label[lid]/float(len(patch_ids)))

    bar_plot(range(1,a_idx+2), [pop_sun, pop_coco], ['SUN Attributes', 'COCO Scene Attributes'], 'pop.png', 'Attributes', 'Percent of Images with Attribute', '', sun_attrs['attributes']) 
        
def worker_stats(blocked = False):

    worker_data = utils.get_all_db_res("select avg(h.time), "\
                                       "stddev(h.time), "\
                                       "count(*) as cnt, "\
                                       "w.id from hit_response h, worker w "\
                                       "where h.worker_id = w.id and "\
                                       "w.employer_id = 1 and "\
                                       "is_blocked = " + str(blocked) + " "\
                                       "group by w.id order by cnt desc")

    ag_msg = ''
    ag_list = [] #[ Avg Time | StdDev Time |  Count Overall | Consensus Agreement Percent [pos, neg] ]
    for item in worker_data:
        wid = item[3]
        worker = Worker.query.get(wid)
        consensus_cnt = filter_workers.consensus_count(wid)
        if consensus_cnt == [0, 0]:
            continue
        ag_list.append([item[0], item[1] if item[1] else 0.0, item[2]] + consensus_cnt)

    x_vals = range(len(ag_list))
    xlabel = 'Workers'
    title = ''
    y_series = [[x[0] for x in ag_list]]
    ylabel = 'Avg HIT time (sec)'
    labels = ['']
    save_name = 'data/worker_avg_time_blocked_%s.png' % str(blocked)
    scatter_plot(x_vals, y_series, labels, save_name, xlabel, ylabel, title)                    
    y_series = [[x[1] for x in ag_list]]
    ylabel = 'StdDev HIT time (sec)'
    labels = ['']
    save_name = 'data/worker_stddev_time_blocked_%s.png' % str(blocked)
    scatter_plot(x_vals, y_series, labels, save_name, xlabel, ylabel, title)                    
    y_series = [[x[2] for x in ag_list]]
    ylabel = 'Cumulative Number of HITs per Worker'
    labels = ['']
    save_name = 'data/worker_num_hits_blocked_%s.png' % str(blocked)
    scatter_plot(x_vals, y_series, labels, save_name, xlabel, ylabel, title)                    
    y_series = [[x[3] for x in ag_list], [x[4] for x in ag_list]]
    ylabel = 'Percent agreement with consensus'
    labels = ['Positive Labels', 'Negative Labels']
    save_name = 'data/worker_consensus_agreement_blocked_%s.png' % str(blocked)
    scatter_plot(x_vals, y_series, labels, save_name, xlabel, ylabel, title)                   

def plot_attr_pop(votes, trend_name, save_name):
    '''
    votes = {(label_id, votes)...}
    '''
    pop = []
    sorted_votes = sorted(votes.items(), key=lambda x: x[0])
    label_names = []
    for item in sorted_votes:
        label_names.append(Label.query.get(item[0]).name)
        pop.append(item[1])
    
    bar_plot(range(1,len(pop)+1), [pop], [trend_name], save_name, 'Attributes', 'Number of votes for Attribute', '', label_names) 

    return pop, label_names
        
        
def bar_plot(x_vals, y_series, labels, save_name, xlabel, ylabel, title, x_ticks = None):
    widthscale = len(x_vals)/4 
    figsize = (widthscale,6) # fig size in inches (width,height)

    fig, ax = plt.subplots(1,1,figsize=figsize)

    bar_width = 3

    opacity = 0.4
    clr = ['r', 'b', 'g']



    for ind, y in enumerate(y_series):
        print labels[ind]
        rects = plt.bar([x*10 + ind*bar_width for x in x_vals], y, bar_width,
                         alpha=opacity,
                         color=clr[ind],
                         label=labels[ind])

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if x_ticks:
        ax.set_xticks([x*10 + ind*bar_width for x in x_vals])
        ax.set_xticklabels(x_ticks, rotation=40, ha='right')
    plt.legend()

    plt.tight_layout()
    plt.grid()
    # plt.show()
    plt.savefig(save_name)
    
    return

def scatter_plot(x_vals, y_series, labels, save_name, xlabel, ylabel, title):
    fig, ax = plt.subplots()
    opacity = 0.4
    clr = ['r', 'b', 'g']



    for ind, y in enumerate(y_series):
        print labels[ind]
        rects = plt.scatter(x_vals, y,
                         alpha=opacity,
                         color=clr[ind],
                         label=labels[ind])

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()

    plt.tight_layout()
    plt.grid()
    # plt.show()
    plt.savefig(save_name)
    
    return


