from sklearn.externals import joblib
from app.models import Label
from mturk import manage_hits



for p in [1, 91, 93, 97]:
    cats = [x.id for x in Label.query.filter(Label.parent_id == p).all()]
    parent = Label.query.get(p)
    print '%s has %d categories' % (parent.name, len(cats))
    label_ids = joblib.load('data/%s_attr_sublist.jbl' % parent.name)
    patch_ids = manage_hits.find_patches(label_ids, [], cats)
    labels = manage_hits.labeled_set(patch_ids, label_ids)
    sname = 'data/%s_exhaustive_labeled_set.jbl' % parent.name
    joblib.dump(labels, sname)
