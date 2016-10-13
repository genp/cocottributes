"""
AWSSigner.py
Copyright (c) 2006 Ansel Halliburton
This is free, public domain software!
"""
import hmac, sha, base64

class AWSSigner:
    """
    Provides HMAC/SHA-1/base64 signatures for Amazon Web Services requests
    Methods:
    __init__
    Parameters:
    secret_key: your AWS 'Secret Access Key'
    sign
    Parameters:
    service
    operation
    timestamp: NOTE: must be in proper format!
    Returns: string
    """
    
    def __init__(self, secret_key):
        self.secret_key = secret_key
    
    def sign(self, service, operation, timestamp):
        h = hmac.new(self.secret_key, str(service) + str(operation) + str(timestamp), sha)
        digest = h.digest()
        return base64.b64encode(digest)
