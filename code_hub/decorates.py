# _*_ coding:utf-8 _*_

"""
@Time: 2022/3/19 6:55 下午
@Author: jingcao
@Email: xinbao.sun@hotmail.com
"""
import time


def calc_time(func):
    def _calc_time(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        ed = time.time()
        print("%s cost time:%s" % (getattr(func, "__name__"), ed - start))
        # l = list(args)
        # print("args: %s" % args)
        return ret

    return _calc_time