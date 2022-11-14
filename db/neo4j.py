# _*_ coding:utf-8 _*_

"""
@Time: 2022/11/11 2:33 下午
@Author: jingcao
@Email: xinbao.sxb@alibaba-inc.com
"""


def write2neo4j(nodes_tgt, rels_tgt, neo4j_path='/Users/peters/Tools/neo4j'):
    """
    如若不想使用graph.db文件，可以再conf/neo4j.conf文件中修改dbms.activate_database的值进行新db的激活挂载
    """
    import os
    neo4j_stop = '{neo4j_path}/neo4j-community-3.5.5/bin/neo4j stop'.format(neo4j_path=neo4j_path)
    db_rm = 'echo "{pwd}"|sudo -S rm -rf {neo4j_path}/neo4j-community-3.5.5/data/databases/graph.db'.format(
        pwd='ssss', neo4j_path=neo4j_path)

    neo4j_import = '{neo4j_path}/neo4j-community-3.5.5/bin/neo4j-admin import \
            --mode=csv \
            --delimiter=$ \
            --database=graph.db \
            --nodes={nodes_tgt} \
            --relationships={rels_tgt} \
            --ignore-extra-columns=true \
            --ignore-missing-nodes=true \
            --ignore-duplicate-nodes=true'.format(neo4j_path=neo4j_path, nodes_tgt=nodes_tgt, rels_tgt=rels_tgt)
    neo4j_start = '{neo4j_path}/neo4j-community-3.5.5/bin/neo4j start'.format(neo4j_path=neo4j_path)
    whole_cmd = ' && '.join([neo4j_stop, db_rm, neo4j_import, neo4j_start])
    os.system(whole_cmd)