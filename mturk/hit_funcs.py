import xml.etree.ElementTree as ET
import requests
from mturk import AWSSigner
import datetime
import config

def get_all_hits(hfile):

    service='AWSMechanicalTurkRequester'
    method='SearchHITs'
    timestamp=datetime.datetime.now().isoformat()
    signer = AWSSigner.AWSSigner(config.mturk_secret_key)
    signature = signer.sign(service, method, timestamp)
    baseurl = 'https://mechanicalturk.amazonaws.com/?Service=%s&AWSAccessKeyId=%s&Operation=%s&Signature=%s&Timestamp=%s&SortProperty=Title&PageSize=100&PageNumber=' % (service, 
                                                                                                                                                                         config.mturk_access_key, 
                                                                                                                                                                         method,
                                                                                                                                                                         signature,
                                                                                                                                                                         timestamp)

    pageind = 1
    r = requests.get('%s%d' % (baseurl, pageind))
    data = ET.XML(r.text)
    hitids = [elem.text for elem in data.iter() if elem.tag == 'HITId']
    allhits = []

    while len(hitids) != 0:
        allhits += hitids
        pageind += 1
        print len(allhits)        
        r = requests.get('%s%d' % (baseurl, pageind))
        data = ET.XML(r.text)
        hitids = [elem.text for elem in data.iter() if elem.tag == 'HITId']


    f = open(hfile, 'w')
    f.write('hitid\n')
    f.write('\n'.join(allhits))
    f.close()
