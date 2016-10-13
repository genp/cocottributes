import os
basedir = os.path.abspath(os.path.dirname(__file__))


###
# App Properties
###

WTF_CSRF_ENABLED = True
SECRET_KEY = ''
SQLALCHEMY_DATABASE_URI = 'postgresql+psycopg2://postgres:...'
SQLALCHEMY_MIGRATE_REPO = os.path.join(basedir, 'db_repository')
SQLALCHEMY_TRACK_MODIFICATIONS = True
log_file = os.path.join(basedir, 'log/server.log')

###
# File Properties
###

img_file_exts = ('.jpg', '.png', '.gif')
mime_dictionary = {
  ".jpg" : "image/jpeg",
  ".jpeg" : "image/jpeg",
  ".gif" : "image/gif",
  ".png" : "image/png"
}

LOCAL_IMAGE_PATH = 'path to coco images'    

###
# ELA Parameters
###

min_patch_area = 50*50    

MAIL_SERVER = 'localhost'
MAIL_SENDER = ''
MAIL_USERNAME = None
MAIL_PASSWORD = None
ADMINS = []
