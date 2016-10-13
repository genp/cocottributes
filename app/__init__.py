#!/usr/bin/env python
from flask import Flask
from flask.ext.sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config.from_object('config')
db = SQLAlchemy(app)

from app import views, models

from config import log_file, ADMINS, MAIL_SERVER, MAIL_USERNAME, MAIL_PASSWORD, MAIL_SENDER

if not app.debug:
    import logging
    from logging.handlers import SMTPHandler
    mail_handler = SMTPHandler(mailhost=MAIL_SERVER, 
                               fromaddr=MAIL_SENDER, 
                               toaddrs=['patterson.genevieve@gmail.com'], 
                               subject='cocottr')
    mail_handler.setLevel(logging.CRITICAL)
    app.logger.addHandler(mail_handler)

    from logging.handlers import RotatingFileHandler
    file_handler = RotatingFileHandler(log_file, 'a', 1 * 1024 * 1024, 10)
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
    app.logger.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.info('cocottributes startup')




