#!/usr/bin/env python
from app import app
import config


# context = ('cocottributes.crt', 'cocottributes.key')
# app.run(host=config.HOST ,port=443, ssl_context=context)#, debug=True)
app.run(host=config.HOST)# ,port=500, debug=True)
