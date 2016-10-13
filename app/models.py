from PIL import Image as PILImage
from io import BytesIO
import numpy as np
import os.path
import requests
import scipy.misc as spm

from app import db
from config import LOCAL_IMAGE_PATH

class Worker(db.Model):
  id = db.Column(db.Integer, primary_key = True)
  username = db.Column(db.String(), index = True, unique = True)
  password = db.Column(db.String(), nullable = False)
  email = db.Column(db.String(120), index=True, unique=True)
  # score on instructional quiz
  quiz_score = db.Column(db.Float(), index = True)
  employer_id = db.Column(db.Integer, db.ForeignKey('employer.id'))
  employer = db.relationship('Employer', backref = db.backref('workers', lazy = 'dynamic'))

  age = db.Column(db.Integer)
  gender = db.Column(db.String())
  education = db.Column(db.String())
  # this is physical location
  location = db.Column(db.String())
  nationality = db.Column(db.String())
  income = db.Column(db.String())
  is_blocked = db.Column(db.Boolean(), default=False)
  on_probation = db.Column(db.Boolean(), default=False)
  ok_until = db.Column(db.Integer, default=30) # stores a number of HITs this worker is approved to do  
  
  def is_authenticated(self):
    return True

  def is_active(self):
    return True

  def is_anonymous(self):
    return False

  def get_id(self):
    return unicode(self.id)

  def __repr__(self):
    return str(self.__dict__)

class Employer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), index=True, unique=True)

    def __repr__(self):
        return '<Employer %r>' % (self.name)


class Image(db.Model):
  id = db.Column(db.Integer, primary_key = True)
  ext = db.Column(db.String())
  mime = db.Column(db.String())
  location = db.Column(db.String())
  # this is test/train/val label
  type = db.Column(db.String())
  url = db.Column(db.String())
  latitude = db.Column(db.Float())
  longitude = db.Column(db.Float())
  width = db.Column(db.Integer)
  height = db.Column(db.Integer)

  def __repr__(self):
    return str(self.__dict__)

  def to_array(self):
    '''
    returns an ndarray of the image contained in this object
    '''
    response = requests.get(self.location)
    img = PILImage.open(BytesIO(response.content))
    return np.array(img)

  
class Patch(db.Model):
  id = db.Column(db.Integer, primary_key = True)
  x = db.Column(db.Float())
  y = db.Column(db.Float())
  width = db.Column(db.Float())
  height = db.Column(db.Float())
  location = db.Column(db.String())
  segmentation = db.Column(db.String())
  area = db.Column(db.Float())
  is_crowd = db. Column(db.Boolean())

  image_id = db.Column(db.Integer, db.ForeignKey('image.id'), index = True)
  image = db.relationship('Image', backref = db.backref('patches', lazy = 'dynamic'))

  def url(self):
    return self.image.url();

  def __repr__(self):
    return str(self.__dict__)

  def cat_id(self):
    cat_id = [x.label.id for x in self.annotations.all() if x.label.parent_id < 102][0]
    return cat_id

  def annotation_vector(self, label_ids, consensus=False):
    '''
    returns 1 x n <number of label_ids> list of the count of 
    positive votes for each label for the given patch
    <ann_vec> contains value of -1 for labels that are not present for this patch
    '''
    ann_vec = -1*np.ones((1, len(label_ids)))
    ann_cnt_vec = np.zeros((1, len(label_ids)))

    anns = [(x.value, x.label_id) for x in self.annotations.filter(Annotation.label_id.in_(label_ids)).all() if x.hit.worker.is_blocked == False]
    for value, lbl_id in anns:
        idx = label_ids.index(lbl_id)
        ann_cnt_vec[0,idx] += 1
        if ann_vec[0,idx] == -1:
            ann_vec[0,idx] = 0
        if value:
            ann_vec[0,idx] += 1

    if consensus:
        for ind, val in enumerate(ann_vec[0][:]):
            if val > -1:
                ann_vec[0,ind] = 1 if np.true_divide(val, ann_cnt_vec[0,ind]) >= 0.5 else 0

    return ann_vec[0], ann_cnt_vec[0]

  def consensus_vec(self, label_ids):
    ann_vec = AnnotationVecMatView.query.filter(AnnotationVecMatView.patch_id == self.id).first()
    if ann_vec == None:
      vec = []            
    else:
      vec = eval(ann_vec.vec)
      lbls = eval(ann_vec.label_ids)
      vec = [vec[lbls.index(i)] for i in label_ids]
    return vec

  def missing_labels(self, label_ids, low_cnt = 0):
    '''
    returns subset of input label_ids for which this patch does not have an annotation
    '''
    ann_vec, ann_cnt_vec = self.annotation_vector(label_ids)
    missing_inds = np.where(ann_cnt_vec <= low_cnt)[1]
    return [label_ids[m] for m in missing_inds]

  def crop(self, savename = None, make_square = False, resize = (), local=False):
    '''
    Saves the bounding box represented by this patch object to file savename
    Options to make the bounding box square and resize to tuple (width, height)
    '''
    if local:
        local_name = os.path.join(LOCAL_IMAGE_PATH, '%s/COCO_%s_%s.jpg' % (self.image.type, self.image.type, '%012d' % self.image.id))
        img = PILImage.open(local_name)
    else:
        response = requests.get(self.image.location)
        img = PILImage.open(BytesIO(response.content))

    top_x = int(self.x)
    top_y = int(self.y)
    bottom_x = int(top_x + self.width)
    bottom_y = int(top_y + self.height)

    if make_square:
      size = max(int(self.width), int(self.height))

      if resize != ():
        if size < max(resize):
          size = max(resize)
      
      remainder_x = int((size-self.width)/2)
      top_x = max(0, top_x - remainder_x)
      bottom_x = top_x + size
      remainder_y = int((size-self.height)/2)
      top_y = max(0, top_y - remainder_y)
      bottom_y = top_y + size

      if bottom_y > img.height:
        error = bottom_y-img.height
        bottom_y = img.height
        bottom_x = bottom_x-error
      if bottom_x > img.width:
        error = bottom_x-img.width
        bottom_x = img.width
        bottom_y = bottom_y-error

    bbox = img.crop((top_x, top_y, bottom_x, bottom_y))

    # always return 3D grayscale images
    if bbox.mode != "RGB":
      tmp = np.zeros(list(bbox.size)+[3])
      tmp[:,:,0] = bbox
      tmp[:,:,1] = bbox
      tmp[:,:,2] = bbox
      bbox = tmp
      
    if len(resize) == 2:
      newshape = resize+[3]
      bbox = spm.imresize(bbox, newshape, 'bicubic')
      
    if savename:
      bbox.save(savename)
    return np.array(bbox)

class Label(db.Model):
  id = db.Column(db.Integer, primary_key = True)
  name = db.Column(db.String())
  parent_id = db.Column(db.Integer, db.ForeignKey('label.id'), index = True)

  defn = db.Column(db.String(length=500))

  def url(self):
    return "/label/"+self.name

  def __repr__(self):
    return str(self.__dict__)

class Word(db.Model):
  id = db.Column(db.Integer, primary_key = True)
  name = db.Column(db.String())

  parent_id = db.Column(db.Integer, db.ForeignKey('label.id'), index = True)
  parent = db.relationship('Label', backref = db.backref('words', lazy = 'dynamic'))

  hit_id = db.Column(db.Integer, db.ForeignKey('hit_response.id'))
  hit = db.relationship('HitResponse', backref = db.backref('words', lazy = 'dynamic'))

  def url(self):
    return "/word/"+self.name

  def __repr__(self):
    return str(self.__dict__)



class Annotation(db.Model):
  id = db.Column(db.BigInteger, primary_key = True)
  value = db.Column(db.Boolean(), nullable=False)
  timestamp = db.Column(db.DateTime(), default=db.func.now())

  patch_id = db.Column(db.Integer, db.ForeignKey('patch.id'), index = True)
  patch = db.relationship('Patch', backref = db.backref('annotations', lazy = 'dynamic'))

  image_id = db.Column(db.Integer, db.ForeignKey('image.id'), index = True)
  image = db.relationship('Image', backref = db.backref('annotations', lazy = 'dynamic'))

  label_id = db.Column(db.Integer, db.ForeignKey('label.id'), nullable=False)
  label = db.relationship('Label', backref = db.backref('annotations', lazy = 'dynamic'))

  hit_id = db.Column(db.Integer, db.ForeignKey('hit_response.id'), nullable=False)
  hit = db.relationship('HitResponse', backref = db.backref('annotations', lazy = 'dynamic'))

  def __repr__(self):
    return str(self.__dict__)

class Feature(db.Model):
  id = db.Column(db.BigInteger, primary_key = True)
  type = db.Column(db.String(), nullable=False)
  location = db.Column(db.String(), nullable=False)# TODO:, unique=True)

  patch_id = db.Column(db.Integer, db.ForeignKey('patch.id'), index = True)
  patch = db.relationship('Patch', backref = db.backref('features', lazy = 'dynamic'))

  image_id = db.Column(db.Integer, db.ForeignKey('image.id'), index = True)
  image = db.relationship('Image', backref = db.backref('features', lazy = 'dynamic'))

  def __repr__(self):
    return str(self.__dict__)

# class Classifier(db.Model):
#   id = db.Column(db.Integer, primary_key = True)
#   name = db.Column(db.String())
#   location = db.Column(db.String())
#   training_state = db.Column(db.String)
#   test_set_percent_change = db.Column(db.Numeric)

#   label_id = db.Column(db.Integer, db.ForeignKey('label.id'), nullable=False)
#   label = db.relationship('Label', backref = db.backref('classifiers', lazy = 'dynamic'))

#   # stores current iteration
#   iteration_id = db.Column(db.Integer, db.ForeignKey('iteration.id'), nullable=False)
#   iteration = db.relationship('Iteration', backref = db.backref('classifier', lazy = 'dynamic'))

#   def url(self):
#     return "/classifier/"+self.name

#   def __repr__(self):
#     return str(self.__dict__)

'''
this is a caching object.
it stores a number value, and associated classifier,
and the examples and active queries for that classifier at iteration.number
'''
class Iteration(db.Model):
  id = db.Column(db.Integer, primary_key = True)
  # this is the positive integer number of this iteration
  number = db.Column(db.Integer, nullable = False)
  score = db.Column(db.String())

  def __repr__(self):
    return str(self.__dict__)


# class Prediction(db.Model):
#   id = db.Column(db.BigInteger, primary_key = True)
#   value = db.Column(db.Float(), index = True, nullable=False)

#   patch_id = db.Column(db.Integer, db.ForeignKey('patch.id'), index = True, nullable=False)
#   patch = db.relationship('Patch', backref = db.backref('predictions', lazy = 'dynamic'))

#   classifier_id = db.Column(db.Integer, db.ForeignKey('classifier.id'), index = True,  nullable=False)
#   classifier = db.relationship('Classifier', backref = db.backref('predictions', lazy = 'dynamic'))

#   iteration_id = db.Column(db.Integer, db.ForeignKey('iteration.id'), nullable=False)
#   iteration = db.relationship('Iteration', backref = db.backref('predictions', lazy = 'dynamic'))

#   def __repr__(self):
#     return str(self.__dict__)

class AnnotationVecMatView(db.Model):
  id = db.Column(db.Integer, primary_key=True)
  patch_id = db.Column(db.Integer)
  label_ids = db.Column(db.String(), nullable=True)
  vec = db.Column(db.String(), nullable=True)
  
  def __repr__(self):
    return str(self.__dict__)

class Jobs(db.Model):
  id = db.Column(db.Integer, primary_key = True)
  cmd = db.Column(db.String(), nullable=False)
  start_time = db.Column(db.DateTime(), default=db.func.now())
  end_time = db.Column(db.DateTime())
  isrunning = db.Column(db.Boolean(), index=True)
  # this is for expected time - 'short', 'long', 'vlong'
  job_type = db.Column(db.String())
  
  def __repr__(self):
    return str(self.__dict__)
  
class Quiz(db.Model):
  id = db.Column(db.Integer, primary_key = True)
  submit_time = db.Column(db.DateTime(), default=db.func.now())
  tp = db.Column(db.Float())
  fp = db.Column(db.Float())
  tn = db.Column(db.Float())
  fn = db.Column(db.Float())
  
  # the user that submitted this quiz
  worker_id = db.Column(db.Integer, db.ForeignKey('worker.id'),  nullable=False)
  worker = db.relationship('Worker', backref = db.backref('quizes', lazy = 'dynamic'))  

  # the job associated with this quiz
  job_id = db.Column(db.Integer, db.ForeignKey('jobs.id'),  nullable=False)
  job = db.relationship('Jobs', backref = db.backref('quizes', lazy = 'dynamic'))  

  def __repr__(self):
    return str(self.__dict__)

class HitResponse(db.Model):
  id = db.Column(db.Integer, primary_key = True)
  # the completion time for all of the patch responses associated with this HIT
  time = db.Column(db.Float(), nullable=False)
  timestamp = db.Column(db.DateTime(), default=db.func.now())
  # the confidence of the labeling user
  confidence = db.Column(db.Integer)
  catch_trial_tp = db.Column(db.Integer)
  catch_trial_fp = db.Column(db.Integer)
  catch_trial_tn = db.Column(db.Integer)
  catch_trial_fn = db.Column(db.Integer)
  score = db.Column(db.String())
  assignment_id = db.Column(db.String())
  mturk_hit_id = db.Column(db.String())

  # the user that submitted this hit
  worker_id = db.Column(db.Integer, db.ForeignKey('worker.id'),  nullable=False)
  worker = db.relationship('Worker', backref = db.backref('hits', lazy = 'dynamic'))

  # the job that this hit is associated with
  job_id = db.Column(db.Integer, db.ForeignKey('jobs.id'),  nullable=False)
  job = db.relationship('Jobs', backref = db.backref('hits', lazy = 'dynamic'))

  def __repr__(self):
    return str(self.__dict__)

class HitQualification(db.Model):
  id = db.Column(db.Integer, primary_key = True)
  qualtypeid = db.Column(db.String())

  # the job associated with this MTurk Qualification
  job_id = db.Column(db.Integer, db.ForeignKey('jobs.id'))
  job = db.relationship('Jobs', backref = db.backref('hit_quals', lazy = 'dynamic'))  

  def __repr__(self):
    return str(self.__dict__)

class HitDetails(db.Model):
  
  id = db.Column(db.BigInteger, primary_key = True)

  timestamp = db.Column(db.DateTime(), default=db.func.now())

  patch_id = db.Column(db.Integer)
  # patch_id = db.Column(db.Integer, db.ForeignKey('patch.id'), index = True)
  # patch = db.relationship('Patch', backref = db.backref('hit_details', lazy = 'dynamic'))

  image_id = db.Column(db.Integer)
  # image_id = db.Column(db.Integer, db.ForeignKey('image.id'), index = True)
  # image = db.relationship('Image', backref = db.backref('hit_details', lazy = 'dynamic'))

  label_id = db.Column(db.Integer)
  # label_id = db.Column(db.Integer, db.ForeignKey('label.id'), nullable=False)
  # label = db.relationship('Label', backref = db.backref('hit_details', lazy = 'dynamic'))

  # ids of submitted hits that contain this (label, patch or image) pair
  hits = db.Column(db.String())
  # length of list in hits
  num_hits = db.Column(db.Integer)

  # the job that defines the hit containing this (label, patch or image) pair
  job_id = db.Column(db.Integer, db.ForeignKey('jobs.id'),  nullable=False)
  job = db.relationship('Jobs', backref = db.backref('hit_details', lazy = 'dynamic'))  

  def __repr__(self):
    return str(self.__dict__)

class Query(db.Model):
  id = db.Column(db.BigInteger, primary_key = True)
  timestamp = db.Column(db.DateTime(), default=db.func.now())
  hit_launched = db.Column(db.Boolean(), default=False)
  patch_id = db.Column(db.Integer)#, db.ForeignKey('patch.id'))
  # patch = db.relationship('Patch', backref = db.backref('queries', lazy = 'dynamic'))

  cat_id = db.Column(db.Integer)#, db.ForeignKey('label.id'))

  label_id = db.Column(db.Integer)#, db.ForeignKey('label.id'))
  # label = db.relationship('Label', backref = db.backref('queries', lazy = 'dynamic'))

  

  def __repr__(self):
    return str(self.__dict__)

class ClassifierScore(db.Model):
  id = db.Column(db.BigInteger, primary_key = True)
  timestamp = db.Column(db.DateTime(), default=db.func.now())
  type = db.Column(db.String(), nullable=False)
  location = db.Column(db.String(), nullable=False)  
  cat_id = db.Column(db.Integer)
  label_id = db.Column(db.Integer)
  num_pos = db.Column(db.Integer)
  num_neg = db.Column(db.Integer)
  train_percent = db.Column(db.Float)
  test_acc = db.Column(db.Float)
  ap = db.Column(db.Float)

  def __repr__(self):
    return str(self.__dict__)
