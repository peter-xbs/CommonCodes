# _*_ coding:utf-8 _*_

"""
@Time: 2022/11/11 2:33 下午
@Author: jingcao
@Email: xinbao.sxb@alibaba-inc.com
"""

import traceback
from ab.utils import logger
from minio import Minio
from minio.error import InvalidResponseError


class OSSEngine(object):
    def __init__(self, endpoint, access_key, secret_key, secure=True):
        self.client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)

    def get_data(self, bucket, dir_name):
        results = []
        msg = 'ok'
        code = 0

        found = self.client.bucket_exists(bucket)
        if not found:
            msg = 'not_found_bucket_error'
            code = 1
            return msg, code, results

        # objects = self.client.list_objects(bucket, prefix=org_id, recursive=True)
        try:
            objects = self.client.list_objects(bucket, prefix=dir_name, recursive=True)
            if not objects:
                msg = 'not_found_object_under{}/{}'.format(bucket, dir_name)
                code = 2
                return msg, code, results

            for obj in objects:
                obj_fh = self.client.get_object(bucket, obj.object_name)
                obj_str = obj_fh.read().decode('utf-8')
                results.append([obj.object_name, obj_str])
        except Exception as e:
            msg = traceback.format_exc()
            code = 3
            return msg, code, results

        else:
            return msg, code, results


def init_oss_conn_engine(oss_cfg):
    oss_elems = oss_cfg.split('$$')
    if len(oss_elems) != 4:
        logger.error('env oss config error!')
        return None
    access_key, secret_key, endpoint, secure = oss_elems
    if secure == 'false':
        secure = False
    else:
        secure = True
    engine = OSSEngine(access_key=access_key, secret_key=secret_key, endpoint=endpoint, secure=secure)
    return engine