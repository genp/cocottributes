#!/usr/bin/env python
# need mapping of qualification ids to quiz job ids

import os

from app import db
from app.models import *
import config


os.environ['JAVA_HOME'] = '/usr'

def grant_qual(qualid, workerid):
    '''
    Grants single qualification to single MTurk worker
    Properties of the qualification, 
    bonuses etc. are store in the qual's property file
    '''
    wrkr = Worker.query.filter(Worker.username == workerid).first()
    if wrkr.is_blocked:
        return
    cwd = os.getcwd()
    in_right_place = os.getcwd() == config.mturk_bin_path
    if not in_right_place:
        os.chdir(config.mturk_bin_path)
    os.system('./assignQualification.sh -qualtypeid %s -workerid %s' % (qualid, workerid)) 
    if not in_right_place:
        os.chdir(cwd)
    

def grant_all_quals(job_id, score=0.8,  timestamp=None):
    '''
    Grant qualification associated with job_id 
    to all workers with a tp score > score 
    after the given time stamp
    '''
    stmt = 'select username from quiz q, worker w where q.worker_id = w.id' \
           ' and job_id = %d and tp >= %f' % (job_id, score)

    if timestamp:
        stmt += ' and q.submit_time > %s' % (str(timestamp))

    db_res = db.engine.execute(stmt).fetchall()
    workers = [x[0] for x in db_res]
    
    qualid = HitQualification.query.filter(HitQualification.job_id==job_id).first().qualtypeid

    for w in workers:
        grant_qual(qualid, w)

    print 'Granted qual %s to %d workers' % (qualid, len(workers))

