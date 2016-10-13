#!/usr/bin/env python
import json

from app import db
from app.models import *

if __name__ == "__main__":

    with open("id2urlid.json", 'r') as f:
        id2urlid = json.load(f)
    for k, v in id2urlid.items():
        az_link = 'https://msvocds.blob.core.windows.net/images/%d_z.jpg' % v
        img = Image.query.get(k)
        img.location = az_link
    db.session.commit()
        

