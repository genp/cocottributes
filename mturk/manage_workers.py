#!/usr/bin/env python
import argparse
import datetime
import os
import requests
import signal
from contextlib import contextmanager

import boto.mturk.connection

from app import db
from app.models import *
from utils import utils
import config

import filter_workers
import mturk_utils

os.environ['JAVA_HOME'] = '/usr'
_host = 'mechanicalturk.amazonaws.com'
_mturk_conn = boto.mturk.connection.MTurkConnection(
    aws_access_key_id = config.mturk_access_key,
    aws_secret_access_key = config.mturk_secret_key,
    host = _host,
    debug = 2
)


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

def notify_workers(worker_ids, subject, msg):
    _mturk_conn.notify_workers(worker_ids, subject, msg)


def block_workers(worker_ids, reject_work, msg = None):
    '''
    Block MTurk workers listed in worker_ids
    Also rejects their work if reject_work is set to True
    '''
    qual_ids = [x[0] for x in utils.get_all_db_res('select distinct qualtypeid from hit_qualification')]

    for wid in worker_ids:
        print 'Attempting to block worker %s ' % wid        
        if reject_work:
            assignment_ids = [x[0] for x in utils.get_all_db_res("select assignment_id from hit_response "\
                                                                 "where assignment_id is not null "\
                                                                 "and assignment_id != '' "\
                                                                 "and assignment_id != 'none' "\
                                                                 "and worker_id = "+wid)]
            aids_string = ','.join(assignment_ids)
            if aids_string != '':                
                rej_cmd = './rejectWork.sh -force -assignment "'+aids_string+'"'
                print rej_cmd
                exec_mturk_cmd(rej_cmd)

        worker = Worker.query.get(wid)
        for qid in qual_ids:
            if msg == None:
                msg = 'Large number of poorly done hits.'
            block_cmd = "./revokeQualification.sh -workerid "+worker.username+" -qualtypeid "+qid+" -reason '"+msg+"'"
            print block_cmd
            exec_mturk_cmd(block_cmd)
        worker.is_blocked = True
        db.session.commit()
        print 'Worker %s all qualifications revoked' % wid
            

def bonus_workers(worker_ids, amount = 0.10, msg = None):
    '''
    Grant Bonus to  MTurk workers listed in worker_ids
    Default value $1.00
    '''
    for wid in worker_ids:
        print 'Attempting to grant bonus to worker %s ' % wid        
        assignment_ids = [x[0] for x in utils.get_all_db_res("select assignment_id, id from hit_response "\
                                                             "where timestamp is not null "\
                                                             "and worker_id = "+wid+" order by timestamp desc")]
        if assignment_ids:          
            if not msg:
                msg = 'Well done hits. Accurate attributes selected.'
            wrkr = Worker.query.get(wid)
            bonus_cmd = './grantBonus.sh -workerid %s -assignment %s '\
                        '-amount %f '\
                        '-reason "%s"' % (wrkr.username, assignment_ids[-1], amount, msg)
            print bonus_cmd
            # TODO catch mturk errors here
            exec_mturk_cmd(bonus_cmd)

            print 'Worker %s granted bonus of %f' % (wid,amount)

        else:
            print 'no assignment ids'
        

def update_workers(worker_ids, amount):
    for wid in worker_ids:
        wrkr = Worker.query.get(wid)
        wrkr.ok_until = amount
        db.session.commit()
        print 'Updated worker %s - ok_until %d hits' % (wid, amount)



def exec_mturk_cmd(cmd):
    cwd = os.getcwd()
    in_right_place = os.getcwd() == config.mturk_bin_path
    if not in_right_place:
        os.chdir(config.mturk_bin_path)
    os.system(cmd) 
    if not in_right_place:
        os.chdir(cwd)

def evaluate_worker(wid, time_interval):
    '''
    Checks for workers that have recently completed HITs.
    Evaluates workers based on inter-annotator agreement over positive labels.
    If worker agrees <= 30% of the time, they are warned and removed from qualification lists. 
    If worker agrees > 30% but <= 60% warned but not removed. Small bonus.
    If worker agrees > 60% larger bonus.
    Input: time_interval - hours elapsed 
    '''

    # get agreement
    hit_cnt, ag, hits_total = filter_workers.consensus_count(wid, time_interval)
    if hit_cnt < 10:
        return ag
    perc = '%.2f' % (100*ag[0])
    # if <= 30% remove qual, send msg
    if ag[0] <= 0.3:
        msg = ('Thank you for working with us. In the last %d hours, '
               'you completed %d HITs for us. Unfortunately, your annotations '
               'rarely agree with other users -- only %s%% of the time. As '
               'a result we would like to thank you for completing a total of %d '
               'HITs for us, but we will be revoking your qualification to '
               'work on our Attribute Annotation HITs. If you feel that you '
               'did not understand the instructions or had particularly tricky HITs '
               'please email us.' % (time_interval, hit_cnt, perc, hits_total))
        amount = 0.05
        print 'Blocking worker %d, %s' % (wid, perc)
        block_workers([str(wid)], reject_work = False, msg = msg)

    # elif > 30% and <= 60% warn, small bonus
    elif ag[0] > 0.3 and ag[0] <= 0.6:
        msg = ('Thank you for working with us. In the last %d hours, '
               'you completed %d HITs for us. Your annotations '
               'agree with other users %s%% of the time. This '
               'is a medium-good performance number. Please try to '
               'look at the images a little more carefully. If you '
               'have an questions about how to do better, please email us. '
               % (time_interval, hit_cnt, perc))
        amount = 0.10

    # else, larger bonus
    else:
        msg = ('Thank you for working with us. In the last %d hours, '
               'you completed %d HITs for us. Your annotations '
               'agree with other users %s%% of the time. This '
               'is a good performance number! Thanks for your hard work!'
               % (time_interval, hit_cnt, perc))
        amount = 0.25
    bonus_workers([str(wid)], amount = amount, msg = msg)

    return ag

def evaluate_workers_daemon(to_addr, wait_hours):
    # try:
    #     with time_limit(wait_hours*60*60): 
    #         while 1:
    #             pass
                
    # except TimeoutException, msg:
    #     print msg
        print 'emailing worker data...'
        worker_data = utils.get_all_db_res("select avg(h.time), "\
                                           "stddev(h.time), "\
                                           "count(*) as cnt, "\
                                           "w.id from hit_response h, worker w "\
                                           "where h.worker_id = w.id and "\
                                           "w.employer_id = 1 and "\
                                           "not is_blocked and "\
                                           "h.timestamp > current_timestamp  - interval '%d hour' "\
                                           "group by w.id order by cnt desc" % wait_hours)

        ag_msg = ''
        ag_list = [' Worker id | Avg Time | StdDev Time | Count %d hrs  |  Count Overall | Consensus Agreement Percent ' % wait_hours]
        for item in worker_data:
            wid = item[3]
            worker = Worker.query.get(wid)
            hits_done_by_worker = utils.get_first_db_res('select count(*) from hit_response where worker_id = %d' % worker.id)
            if hits_done_by_worker > worker.ok_until:
                
                res = evaluate_worker(wid, wait_hours)
                ag_list.append('%d  |  %.2f  |  %.2f  |  %d  |  %d  |  %s' % (wid, 
                                                                              item[0], item[1] if item[1] else 0.0, item[2], hits_done_by_worker, 
                                                                              '%.4f  ,  %.4f' % tuple(res)))
        ag_msg = '\n'.join(ag_list)

        mail_msg =  ag_msg
        
        utils.email_notify(message=mail_msg, subject="[cocottributes] Eval Workers "+str(datetime.datetime.now()), to_addr=to_addr)

        evaluate_workers_daemon(to_addr, wait_hours)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--workers", nargs='+')
    parser.add_argument("--reject", action="store_true", help="also reject their work")
    parser.add_argument("--bonus", action="store_true", help="grant these workers bonus")
    parser.add_argument("--update", action="store_true", help="update ok_until number of hits param for workers")
    parser.add_argument("--amount", type=float, help="bonus amount - default $0.10, or update ok_until amount - no default")
    parser.add_argument("--msg", type=str, help="bonus message")
    parser.add_argument("-m", "--mail_addr", type=str, help='email address to send updates to')
    parser.add_argument("--hours", type=int, help='time between email updates (hours)')
    args = parser.parse_args()

    if args.workers:
        if args.bonus:
            bonus_workers(args.workers, args.amount, args.msg)
        elif args.update:
            update_workers(args.workers, args.amount)
        else:
            block_workers(args.workers, args.reject, args.msg)
    else:
        evaluate_workers_daemon(args.mail_addr, args.hours)
