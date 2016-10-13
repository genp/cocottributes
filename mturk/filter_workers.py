#!/usr/bin/env python
import os
import argparse
import datetime
import signal
from contextlib import contextmanager

from app import db
from app.models import *
from utils import utils

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException, "Timed out!"
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def filter_workers_daemon(to_addr, wait_sec):
    try:
        with time_limit(wait_sec): 
            while 1:
                pass
                
    except TimeoutException, msg:
        print msg
        print 'emailing worker data...'
        worker_data = utils.get_all_db_res("select avg(h.time), "\
                                           "stddev(h.time), "\
                                           "count(*) as cnt, "\
                                           "w.id from hit_response h, worker w "\
                                           "where h.worker_id = w.id and "\
                                           "w.employer_id = 1 and "\
                                           "not is_blocked and "\
                                           "h.timestamp > current_timestamp  - interval '24 hour' "\
                                           "group by w.id order by cnt desc")

        ag_msg = ''
        ag_list = [' Worker id | Avg Time | StdDev Time | Count 24 hrs  |  Count Overall | Consensus Agreement Percent ']
        for item in worker_data:
            wid = item[3]
            worker = Worker.query.get(wid)
            hits_done_by_worker = utils.get_first_db_res('select count(*) from hit_response where worker_id = %d' % worker.id)
            if hits_done_by_worker > worker.ok_until:
                time_interval_hours = wait_sec/60/60 if wait_sec/60/60 > 0 else 1
                consensus_cnt = consensus_count(wid, time_interval_hours)
                # if consensus_cnt[0] < 0.05 : # if the worker agrees with the consensus of positive labels less than 5% of the time
                print consensus_cnt
                ag_list.append('%d  |  %.2f  |  %.2f  |  %d  |  %d  |  %s' % (wid, 
                                                                              item[0], item[1] if item[1] else 0.0, item[2], hits_done_by_worker, 
                                                                              '%.4f  ,  %.4f' % tuple(consensus_cnt[1])))
        ag_msg = '\n'.join(ag_list)

        mail_msg =  ag_msg
        
        utils.email_notify(message=mail_msg, subject="[cocottributes] Filter Workers "+str(datetime.datetime.now()), to_addr=to_addr)

        filter_workers_daemon(to_addr, wait_sec)
        
def consensus_count(wid, time_interval = '24', good_job_type = '', bad_job_type = '', print_on = False):
    '''
    Counts number of times worker wid differs from the consensus on an annotation
    wid is an int for primary key of worker in database
    time_interval is in hours
    '''
    stmt = "select count(*) from hit_response h, worker w, jobs j where h.worker_id = w.id and h.job_id = j.id and w.id = %s " % str(wid)

    # stmt += "and j.job_type not like 'scene%%' "
    if good_job_type != '':
        stmt += "and j.job_type like %s " % good_job_type
    if bad_job_type != '':
        stmt += "and j.job_type not like %s " % bad_job_type

    hits_total = utils.get_first_db_res(stmt)

    stmt += " and timestamp > current_timestamp - interval '%d hour' " % time_interval
    hits_cnt = utils.get_first_db_res(stmt)
    hit_limit = 500
    if hits_cnt > hit_limit:
        print 'Worker %d did %d hits, only calculating agreement on last %d hits' % (wid, hits_cnt, hit_limit)
        # return [ 0.0, 0.0]
    stmt = "select a1.patch_id, a1.label_id, a1.value, a2.value "\
           "from annotation a1, annotation a2, hit_response h1, hit_response h2, jobs j "\
           "where a1.hit_id = h1.id "\
           "and a1.label_id = a2.label_id and a1.patch_id = a2.patch_id "\
           "and h1.worker_id = "+str(wid)+" "\
           "and a1.id <> a2.id and a2.hit_id = h2.id "\
           "and h1.job_id = j.id "
    stmt += "and h1.timestamp > current_timestamp - interval '%d hour' " % time_interval
    stmt += "and h2.timestamp > current_timestamp - interval '%d hour' " % time_interval
    # and j.job_type not like 'scene%%' 
    if good_job_type != '':
        stmt += "and j.job_type like %s " % good_job_type
    if bad_job_type != '':
        stmt += "and j.job_type not like %s " % bad_job_type
    stmt += "order by a1.patch_id, a1.label_id limit %d" % (hit_limit)
    votes = utils.get_all_db_res(stmt)
    pid = 0
    lid = 0
    diff_cnt = 0
    pos_cnt = 0
    tot_cnt = 0
    agree_consensus_cnt = [0,0] # [pos count, neg count]
    consensus_cnt = [0,0] # [pos count, neg count]

    for item in votes:
        cur_pid, cur_lid, wid_vote, other_vote = item
        if print_on:
            print item
        if cur_pid != pid or cur_lid != lid:
            if tot_cnt > 0 and float(pos_cnt)/tot_cnt >= 0.5:
                consensus_cnt[0] += 1
            elif tot_cnt > 0 and float(pos_cnt)/tot_cnt < 0.5:
                consensus_cnt[1] += 1

            if tot_cnt > 0 and float(diff_cnt)/tot_cnt < 0.5:
                if float(pos_cnt)/tot_cnt >= 0.5:
                    agree_consensus_cnt[0] += 1
                else:
                    agree_consensus_cnt[1] += 1

            pid = cur_pid
            lid = cur_lid
            diff_cnt = 0
            pos_cnt = 1 if wid_vote == True else 0
            tot_cnt = 0

        tot_cnt += 1
        if wid_vote != other_vote:
            if print_on:
                print 'diff vote!'
            diff_cnt += 1

        if other_vote == True:
            pos_cnt +=1
        

    ag_percent = [float(agree_consensus_cnt[i])/(consensus_cnt[i] + 0.0001) for i in range(2)]
    print 'Worker %d (%d hits in last %d hours) agreed with consensus %s / %s times (%s) [pos, neg]' % (wid, hits_cnt, time_interval, str(agree_consensus_cnt), str(consensus_cnt), str(ag_percent))

    return hits_cnt, ag_percent, hits_total
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mail_addr", type=str, help='email address to send updates to')
    parser.add_argument("-w", "--wait_sec", type=int, help='time between email updates')
    parser.add_argument("-c", "--consensus", action="store_true", help='calc consensus agreement, need worker id to also be specified')
    parser.add_argument("--worker", type=int, help='worker id to check')
    args = parser.parse_args()

    if args.consensus:
        consensus_count(args.worker, print_on = True)
    else:
        filter_workers_daemon(args.mail_addr, args.wait_sec)
