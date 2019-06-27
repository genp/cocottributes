# Update Feb 2019: 
New pytorch models! Significant improvement over 2016 baseline network. See updates in pytorch dir. 

# COCO Attributes 
## annotation server and attribute classification code
Code used to create COCO Attributes dataset and experiments in the associated ECCV 2016 paper. 

Before using this repository, create a config.py in the top-level
directory. Use config_example.py as a guide.

# Requirements:
Python requirements are listed in requirements.txt. To install use:
> pip install -r requirements.txt

# Additional requirements include:
* Nginx (example config file in nginx.conf.bak)
* Postgres
* Caffe (for running classification experiements, not needed for annotation server)

# COCO Attributes dataset
The current version of this dataset (pickled for Python 3) is located at `./data/cocottributes_eccv_version.pkl`, synced with [git lfs](https://git-lfs.github.com/)

To download the current version of this dataset (for Python 2) go to:
http://cs.brown.edu/~gmpatter/cocottributes.html

# Instructions for starting the server
sudo service nginx start
uwsgi --socket 127.0.0.1:8081 -w wsgi:app

# Instructions for starting mturk hits

manage_hits.py
    '''
    returns list of job_ids that have less than 3 hits by trusted workers
    job_id occur multiple times if missing multiple hits
    '''
    get_missing_hits(job_type):
    
    launch_missing_hits(job_type, task_file, mturk_rel_path)

ela_hits.py
    # Note: this module is called by views.py, not by the requester directly in normal operation
    '''
    this is the number of questions permitted in the ELA
    '''
    NUMQ = 40 (set this variable in config.py for max number of attributes to annotate per object)
    '''
    func for adding (patch, attribute) question to Query table, check if enough queries to launch new hit
    for relaunching hits that already have labels
    '''
    schedule_next_query(patch_id, cat_id)

make_task.py
    '''
    Batches patch_ids into groups of 10 and attr_ids into groups of 20 
    Makes set of HITs
    allimgs=True for binary attribute labeling tasks with one image per query
    allimgs=False for multiple attribute annotation
    Output: list of job_ids created
    '''
    make_tasks(patch_ids, attr_ids, task_label, job_type='annotation', allimgs=False)

views.py:
    '''
    server function that runs when a form from the 'allimgs' page is submitted
    these are assumed to be ela tasks, and the ela function is queried after the annotations are logged
    '''
    allimgs_post(self, id)
