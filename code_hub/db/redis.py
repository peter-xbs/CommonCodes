# _*_ coding:utf-8 _*_

"""
@Time: 2022/11/11 3:03 下午
@Author: jingcao
@Email: xinbao.sun@hotmail.com
"""
"""
使用redis做缓存，这里模拟一个web接口缓存的例子, todo 待进一步实践
"""

import re
import functools
import json
import time
from functools import wraps

import redis
from redis.exceptions import ConnectionError, ResponseError, TimeoutError

func_run_stat_dict = {

}
try:
    import line_profiler
except:
    pass


def fun_run_time(func):
    @wraps(func)
    def inner(*args, **kwargs):
        s_time = time.time()
        # profiler = line_profiler.LineProfiler(func)
        # profiler.enable()
        ret = func(*args, **kwargs)
        # profiler.disable()
        # profiler.print_stats()
        e_time = time.time()
        func_name = f'{func.__module__}-{func.__class__}-{func.__name__}'
        func_stat = func_run_stat_dict.get(func_name, {'call_count': 0, 'total_cost_time': 0.0})
        func_stat['call_count'] += 1
        func_stat['total_cost_time'] += e_time - s_time
        func_stat['total_cost_time'] += e_time - s_time
        func_run_stat_dict[func_name] = func_stat
        return ret

    return inner


config_dict = {}  # todo 提供redis连接信息

if 'redis_passwd' in config_dict:
    pool = redis.ConnectionPool(host=config_dict['redis_url'], port=config_dict['redis_port'],
                                password=config_dict['redis_passwd'], username=config_dict['redis_usr'])
else:
    pool = redis.ConnectionPool(host=config_dict['redis_url'], port=config_dict['redis_port'])

redis_cli = redis.Redis(connection_pool=pool)


def redis_cache(func):
    @functools.wraps(func)  # 为了保留原函数的属性，因为被装饰的函数对外暴露的是装饰器的属性
    def wrapper(*args, **kargs):

        start_time = time.time()
        _key = 'function-name:{},args:{},kargs:{}'.format(func.__name__, args, kargs)  # 定义key的形式：函数名加与参数组成唯一的key
        _key = re.sub('<.+?>', '', _key)
        try:
            result = redis_cli.get(_key)
            if result:  # redis查找到对应的key，直接返回结果
                result = json.loads(result)
                print(type(result))
                print('redis find:{},result:{}'.format(_key, result))
            else:  # redis没有查找到对应key，查询执行函数，查询mysql
                print('redis not find:{}'.format(_key))
                result = func(*args, **kargs)
                redis_cli.setex(name=_key, value=json.dumps(result),
                                time=config_dict['redis_expire_time'])  # 将mysql结果写入redis,并设置过期时间 单位s
        except (ConnectionError, TimeoutError, ResponseError) as e:
            import traceback
            traceback.print_exc()
            result = func(*args, **kargs)

        print("final result:{}".format(result))
        end_time = time.time() - start_time
        print("Total time of this query:{}".format(end_time))
        return result

    return wrapper


if __name__ == '__main__':
    @redis_cache
    def mysql_dispose(name, age):
        time.sleep(2)
        result = {'name:': name, 'age': age}
        print('mysql-result:{}'.format(result))
        return (result)


    @redis_cache
    def json_func(json_object, oo=None):
        time.sleep(2)
        print('mysql-result:{}'.format(json_object))
        return json_object


    # mysql_dispose('zz3', 45)
    json_func({'key': 0, 'key2': [0, 1, 3, 4, 5], 'o': object(), 'key3': 12323, 'o2': object()}, object())
    json_func({'key': 0, 'key2': [0, 1, 3, 4, 5]})
