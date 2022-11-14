# _*_ coding:utf-8 _*_

"""
@Time: 2022/11/11 2:33 下午
@Author: jingcao
@Email: xinbao.sun@hotmail.com
"""
import json

class ElasticSearchEngine(object):
    def __init__(self, index_name="info_icd10"):
        from elasticsearch import Elasticsearch
        self.es = Elasticsearch(['es-cn-2r42a4k3x000ks3to.public.elasticsearch.aliyuncs.com'],
                                http_auth=('elastic', 'CDSS_es_search'), port=9200, use_ssl=False)
        self.index_name = index_name

    def creat_index(self, json_list):
        import tqdm
        import json
        from elasticsearch import helpers
        q_set = set()
        result = self.es.indices.create(index=self.index_name, ignore=400)
        print(result)

        mapping = {
            'properties': {
                'query': {
                    'type': 'text',
                    "analyzer": "simple",
                    "search_analyzer": "simple",
                    'index': True
                },
                'info': {
                    'type': 'text',
                },
            }
        }

        self.es.indices.put_mapping(index=self.index_name, body=mapping)
        nums = len(json_list)
        chunk_size = 1000
        pbar = tqdm.trange(0, nums, chunk_size)
        for begin in pbar:
            doc_chunk = json_list[begin: begin + chunk_size]
            bulk_data = []
            for json_object in doc_chunk:
                feature = json_object["entity_term"]
                query = list(feature)
                query = ' '.join(query)
                if query in q_set:
                    continue
                q_set.add(query)
                try:
                    cur = {
                        "_index": self.index_name,
                        "_source": {
                            "query": query,
                            "info": json.dumps(json_object, ensure_ascii=False)
                        }}
                except:
                    print()
                bulk_data.append(cur)

            helpers.bulk(self.es, bulk_data)

    def del_index(self):
        result = self.es.indices.delete(index=self.index_name, ignore=[400, 404])
        print(result)

    def search(self, query, max_recall=500):
        result = {}
        query = " ".join(list(query))
        query = {
            "query": {
                "match": {
                    "query": query
                }
            }
        }
        cur_result = self.es.search(index=self.index_name, body=query, size=max_recall)
        query_results = cur_result['hits']["hits"]

        for query_result in query_results:
            json_object = {}
            json_object["name"] = " ".join(query_result["_source"]["query"].split(" "))
            json_object["id"] = query_result["_id"]
            json_object["info"] = query_result["_source"]["info"]
            json_object["score"] = query_result["_score"]
            id = json_object["id"]
            score = json_object["score"]
            if id not in result:
                result[id] = json_object
            else:
                if result[id]["score"] < score:
                    result[id]["score"] = score
        result = sorted(list(result.values()), key=lambda x: x["score"], reverse=True)

        for r in result:
            r['info'] = json.loads(r['info'])
        return result