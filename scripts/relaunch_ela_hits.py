#!/usr/bin/env python
import logging

from app import db
from mturk import ela_hits

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Select 5k person patches that have some labels but than the ELA limit 
#stmt = "select patch_id from (select patch_id, count(*) as nh from annotation where patch_id in (select patch_id from annotation where label_id = 1) group by patch_id) as foo where nh > 3 and nh < 10 order by nh desc limit 5000"
stmt = "select patch_id from (select patch_id, count(distinct label_id) as cnt from annotation where patch_id in (select patch_id from annotation where label_id = 1) and (label_id > 407 or label_id = 1) group by patch_id) as foo where cnt = 2 limit 5000;"
patch_ids = [x[0] for x in db.engine.execute(stmt)]
logging.debug('num_patches %d' % len(patch_ids))
logging.debug('first patch_id %d' % patch_ids[0])
for patch_id in patch_ids:
    ela_hits.schedule_next_query(patch_id, cat_id = 1)
