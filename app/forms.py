from flask.ext.wtf import Form
from wtforms import StringField, BooleanField, HiddenField
from wtforms.validators import DataRequired

class LoginForm(Form):
    openid = StringField('openid', validators=[DataRequired()])
    remember_me = BooleanField('remember_me', default=False)

class DiffForm(Form):
    '''
    Form contains text responses (list of words), intended for mturk task of differentiating objects
    '''
    resp = HiddenField('resp', validators = [DataRequired()])
    time = HiddenField('time', validators = [DataRequired()])
    worker = HiddenField('worker', validators = [DataRequired()])
    location = HiddenField('location')
    nationality = HiddenField('nationality')

class AnnForm(Form):
    '''
    Form contains attribute annotations
    '''
    resp = HiddenField('resp', validators = [DataRequired()])
    time = HiddenField('time', validators = [DataRequired()])
    worker = HiddenField('worker', validators = [DataRequired()])
    location = HiddenField('location')
    nationality = HiddenField('nationality')
    assignment_id = HiddenField('assignment_id')
    hit_id = HiddenField('hit_id')
