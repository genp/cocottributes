import fnmatch
import os
import json

from app.models import *
from app import db
from utils import *

def add_words():
    print 'h2'
    fdir = 'data/'
    for file in reversed(os.listdir(fdir)):
        if fnmatch.fnmatch(file, '*_word_cnt.json'):
            print file
            if 'caption' in file:
                hit_id = 1891
            else:
                hit_id = 1890
            wlist = []
            stoplist = []
            with open(os.path.join(fdir, file), 'r') as f:
                words = json.load(f)
            print words.keys()
            obj = words.keys()[0]
            objname = obj.replace('_', ' ')
            
            try:
                label = Label.query.filter(Label.name == objname).all()[0]
            except IndexError, e:
                continue
            print label

            use_obj = raw_input('label this obj: '+objname+' ?')
            if use_obj == 's':
                continue
            elif use_obj == 'skip':
                continue
            elif use_obj == 'break':
                break

            for wtype in words[obj].keys():
                for idx, item in enumerate(words[obj][wtype][:100]):
                    print '***'
                    print item
                    item = [item[0].replace("'", "''"), item[1]]
                    # check if item[0] is in stop_words
                    query = "select id from %s where name = '%s'" % ('stop_word', item[0])
                    if get_all_db_res(query) == []:
                        query = "select id from %s where name = '%s' or name = '%s' or name = '%s'" % ('word', item[0], item[0]+'ing', item[0][:-1]+'ing')
                        if get_all_db_res(query) != []:
                            continue
                    else:
                        continue

                    
                    resp = raw_input(str(idx)+': '+objname+' : '+str(item)+' ?')
                    print resp
                    if resp == 'a':
                        wlist.append(item[0])
                    elif resp == 's' or resp == '':
                        stoplist.append(item[0])
                    elif resp == 'keep': # for if the word is bad for this obj but not bad generally
                        continue
                    elif resp == 'skip':
                        stoplist += [x[0] for x in words[obj][wtype][idx:100]]
                        break
                    else:
                        wlist.append(resp)

            stmt = 'insert into word values '
            for w in wlist:
                stmt += "(default, '%s', %d, %d), " % (w, label.id, hit_id)
            stmt = stmt[:-2]
            print stmt
            use = raw_input('Good?')
            if use == 'a' or use == 'y':                
                db.engine.execute(stmt)
                
            stmt = 'insert into stop_word values '
            for w in stoplist:
                stmt += "(default, '%s'), " % (w)
                
            stmt = stmt[:-2]
            print stmt
            use = raw_input('Good?')
            if use == 'a' or use == 'y':                
                db.engine.execute(stmt)

    print 'Done!!!'

if __name__ == "__main__":
    print 'h1'
    add_words()
