# _*_ coding:utf-8 _*_

"""
@Time: 2022/3/19 6:43 下午
@Author: jingcao
@Email: xinbao.sun@hotmail.com
"""


def create_id():
    import time, hashlib
    m = hashlib.md5()
    m.update(bytes(str(time.clock()), encoding='utf-8'))
    return m.hexdigest()


def create_id_by_uuid(identifier):
    import uuid
    if not isinstance(identifier, str):
        identifier = str(identifier)
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, identifier))