import xml.etree.ElementTree as ET
import requests
from mturk import AWSSigner
import datetime

import config

def url_stub(method):
    service='AWSMechanicalTurkRequester'
    timestamp=datetime.datetime.now().isoformat()
    signer = AWSSigner.AWSSigner(config.mturk_secret_key)
    signature = signer.sign(service, method, timestamp)
    baseurl = 'https://mechanicalturk.amazonaws.com/?'\
              'Service=%s&AWSAccessKeyId=%s&'\
              'Operation=%s&'\
              'Signature=%s&'\
              'Timestamp=%s&' % (service, config.mturk_access_key, method, signature, timestamp)
    return baseurl

def get_all_hits(hfile):
    method='SearchHITs'
    baseurl = url_stub(method) + 'SortProperty=Title&PageSize=100&PageNumber='

    while len(hitids) != 0:
        
        r = requests.get('%s%d' % (baseurl, pageind))
        data = ET.XML(r.text)
        hitids = [elem.text for elem in data.iter() if elem.tag == 'HITId']
        allhits += hitids
        pageind += 1

    f = open(hfile, 'w')
    f.write('hitid\n')
    f.write('\n'.join(allhits))
    f.close()

    return allhits
