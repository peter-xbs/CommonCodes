# _*_ coding:utf-8 _*_

"""
@Time: 2022/5/5 9:29 下午
@Author: jingcao
@Email: xinbao.sxb@alibaba-inc.com
写入数据库的若干方法
"""
import os
import json
import pymysql
from hub.decorates import calc_time

import traceback
from ab.utils import logger
# import mysql.connector.pooling
from mysql.connector import pooling


@calc_time
def write2db(db="local", ):
    if db == "local":
        conn = pymysql.connect(
            host="localhost",
            port=3306,
            user="root",
            password="petersssss",
            database="medical_kg",
            charset="utf8",
            local_infile=True  # 设置为TRUE方可从LOCAL上传文件
        )
    elif db == "rds":
        conn = pymysql.connect(
            host="rm-bp1g76bc322l4b4grfo.mysql.rds.aliyuncs.com",
            port=3306,
            user="cdss_kg",
            password="ML@cdss_kg",
            database="medical_kg",
            charset="utf8",
            local_infile=True  # 设置为TRUE方可从LOCAL上传文件
        )
    else:
        raise Exception("wrong db arguments, must in [local, rds]...")
    cursor = conn.cursor()
    # create_table
    sql = 'USE medical_kg'
    cursor.execute(sql)

    sql = "DROP TABLE if exists nodes;"
    cursor.execute(sql)
    sql = """CREATE TABLE if not exists nodes (
                id INT(11) PRIMARY KEY AUTO_INCREMENT,
                entity_id VARCHAR(255),
                entity_name VARCHAR(255),
                entity_tag VARCHAR(255),
                entity_attrs TEXT,
                source TEXT)"""
    cursor.execute(sql)

    # load_data1
    sql = "load data local infile '/Users/peters/PycharmProjects/KnowledgeGraphConstruct/DB/nodes.csv' replace into TABLE nodes character set utf8 fields terminated by '$' lines terminated by '\\n' (entity_id, entity_name, entity_tag, entity_attrs, source);"
    cursor.execute(sql)
    # 数据库表更新需本句sql
    # 创建关系表
    sql = "DROP TABLE if exists relations;"
    cursor.execute(sql)
    sql = """CREATE TABLE if not exists relations (
                id INT(11) PRIMARY KEY AUTO_INCREMENT,
                src_ent_id VARCHAR(255),
                property VARCHAR(255),
                tgt_ent_id VARCHAR(255),
                tgt_ent_group VARCHAR(255)
                )
                """
    cursor.execute(sql)
    # 载入关系数据
    sql = "load data local infile '/Users/peters/PycharmProjects/KnowledgeGraphConstruct/DB/relations.csv' replace into TABLE relations character set utf8 fields terminated by '$' lines terminated by '\\n' (src_ent_id, property, tgt_ent_id, tgt_ent_group);"
    cursor.execute(sql)
    # 为关系表建立索引
    sql = "CREATE INDEX node_ent_id ON nodes (entity_id)"
    cursor.execute(sql)
    sql = "CREATE INDEX node_ent_name ON nodes (entity_name)"
    cursor.execute(sql)
    sql = "CREATE INDEX node_ent_tag ON nodes (entity_tag)"
    cursor.execute(sql)
    sql = "CREATE INDEX relation_ent_id ON relations (src_ent_id)"
    cursor.execute(sql)
    sql = "CREATE INDEX relation_propery ON relations (property)"
    cursor.execute(sql)

    # 查询指定表大小
    # query_table_size(cursor, 'medical_kg', 'nodes')
    # query_table_size(cursor, 'medical_kg', 'relations')
    # # 查询索引大小
    # query_index_size(cursor, 'medical_kg', 'nodes')
    # query_index_size(cursor, 'medical_kg', 'relations')
    conn.commit()
    cursor.close()
    conn.close()

class MySQLPool(object):
    """
    create a pool when connect mysql, which will decrease the time spent in
    request connection, create connection and close connection.
    """
    def __init__(self, host="172.0.0.1", port="3306", user="root",
                 password="123456", database="test", pool_name="mypool",
                 pool_size=3):
        res = {}
        self._host = host
        self._port = port
        self._user = user
        self._password = password
        self._database = database

        res["host"] = self._host
        res["port"] = self._port
        res["user"] = self._user
        res["password"] = self._password
        res["database"] = self._database
        self.dbconfig = res
        self.pool = self.create_pool(pool_name=pool_name, pool_size=pool_size)

    def create_pool(self, pool_name="mypool", pool_size=3):
        """
        Create a connection pool, after created, the request of connecting
        MySQL could get a connection from this pool instead of request to
        create a connection.
        :param pool_name: the name of pool, default is "mypool"
        :param pool_size: the size of pool, default is 3
        :return: connection pool
        """
        pool = pooling.MySQLConnectionPool(
            pool_name=pool_name,
            pool_size=pool_size,
            pool_reset_session=True,
            **self.dbconfig)
        return pool

    def close(self, conn, cursor):
        """
        A method used to close connection of mysql.
        :param conn:
        :param cursor:
        :return:
        """
        cursor.close()
        conn.close()
    def execute(self, sql, args=None, commit=False):
        """
        Execute a sql, it could be with args and with out args. The usage is
        similar with execute() function in module pymysql.
        :param sql: sql clause
        :param args: args need by sql clause
        :param commit: whether to commit
        :return: if commit, return None, else, return result
        """
        # get connection form connection pool instead of create one.
        conn = self.pool.get_connection()
        cursor = conn.cursor()
        code = 0
        if args:
            cursor.execute(sql, args)
        else:
            cursor.execute(sql)
        if commit is True:
            conn.commit()
            self.close(conn, cursor)
            return code, None
        else:
            res = []
            try:
                r = cursor.fetchone()
                while r:
                    res.append(r)
                    r = cursor.fetchone()
            except Exception as e:
                err = traceback.format_exc()
                logger.error(err)
                code = 1
            # res = cursor.fetchall()
            self.close(conn, cursor)
            return code, res

    def executemany(self, sql, args, commit=False):
        """
        Execute with many args. Similar with executemany() function in pymysql.
        args should be a sequence.
        :param sql: sql clause
        :param args: args
        :param commit: commit or not.
        :return: if commit, return None, else, return result
        """
        # get connection form connection pool instead of create one.
        conn = self.pool.get_connection()
        cursor = conn.cursor()
        cursor.executemany(sql, args)
        if commit is True:
            conn.commit()
            self.close(conn, cursor)
            return None
        else:
            res = cursor.fetchall()
            self.close(conn, cursor)
            return res


def init_db_conn_engine(db_cfg, system='Linux'):
    import re
    db_elems = re.split('[:/]', db_cfg)
    if len(db_elems) != 4:
        logger.error("env db config error!")
        return None
    user, pswd_addr, port, db_name = db_elems
    pswd_addr = re.split('@', pswd_addr)
    if len(pswd_addr) == 2:
        pswd, addr = pswd_addr
    elif len(pswd_addr) > 2:
        pswd = '@'.join(pswd_addr[:-1])
        addr = pswd_addr[-1]
    else:
        return None
    config_dict = {'host': addr, 'user': user, 'password': pswd, 'database': db_name, 'port': int(port)}
    logger.error(config_dict)
    engine = MySQLPool(**config_dict)
    return engine
