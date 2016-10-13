#!/usr/bin/env python
import argparse
import wikipedia

from app import db
from app.models import *

def create_labels(input_file):
    with open(input_file, 'r') as f:
        new_lbls = f.read().splitlines()

    parent_name = raw_input('What is the label parent name for these new labels? ')
    parent_id = Label.query.filter(Label.name == parent_name).first().id
    print 'Parent id: %d' % parent_id 
    is_same_parent = input('True/False do all these labels have same parent? ')
        
    aux_defn = {}
    aux_defn['indoor'] = " The picture should be taken from inside."
    aux_defn['outdoor']  = " The picture should be taken from outside."

    for lbl in new_lbls:
        entry = try_wikipedia(lbl.split(' (')[0])
        
        for aux in aux_defn.keys():
            if aux in lbl:
                entry += aux_defn[aux]

        print '%s : %s' % (lbl, entry)

        new_defn = raw_input('Change defn? ') # if the user enters nothing, var entry will be used
        if new_defn:
           entry = new_defn
        
        if not is_same_parent:
            parent_name = raw_input('What is the label parent name for this label (enter same for same as %s)? ' % parent_name)
            if not parent_name == 'same':
                parent_id = Label.query.filter(Label.name == parent_name).first().id

        db_lbl = Label()
        db_lbl.name = lbl
        db_lbl.parent_id = parent_id
        db_lbl.defn = entry
        db.session.add(db_lbl)
        db.session.commit()

def try_wikipedia(item):
    try:
        entry = wikipedia.summary(item, sentences=1)
    except wikipedia.exceptions.DisambiguationError as e:
        print e.options
        which = input('Please enter index of correct entry (0-based): ')
        entry = try_wikipedia(e.options[which])
    except wikipedia.exceptions.PageError as e:
        entry = "couldn't find defn"

    return entry

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='add stuff to cocottributes database')
    parser.add_argument('input_file', type=str, 
                        help='file with list of labels to add')


    args = parser.parse_args()
    create_labels(args.input_file)
