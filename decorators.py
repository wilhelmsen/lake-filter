# coding: utf-8
import datetime
import logging
LOG = logging.getLogger(__name__)

def timer(f):
    def wrapper(*args, **kw):
        t = datetime.datetime.now()
        res = f(*args, **kw)
        LOG.debug("%s took %0.3fs." % (f.__name__, (
                    datetime.datetime.now() - t).total_seconds()))
        return res
    return wrapper


