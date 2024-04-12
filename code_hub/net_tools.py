# _*_ coding:utf-8 _*_

def get_ip():
    import os
    r = os.popen("hostname -I|awk '{print $1}'")
    r = r.read().strip()
    return r
