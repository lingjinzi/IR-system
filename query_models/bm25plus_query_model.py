import time
from utilities import utils
from utilities import Constants
import statistics


class BM25PLUSQueryModel:
    def __init__(self, query_term_frequency, tf_inverted_index, doc_info, b=0.75, k1=1.2, k3=10):
        self.query_term_frequency = query_term_frequency
        self.tf_inverted_index = tf_inverted_index
        self.doc_info = doc_info
        self.bm25plus_score = dict.fromkeys(self.doc_info.keys(), 0.0)
        self.ranked_doc_ids = list()

        # BM25+算法参数
        self.b = b
        self.k1 = k1
        self.k3 = k3

    def execute_query(self):
        start_time = time.time()

        # 参数计算
        avg_dl = statistics.mean(self.doc_info.values())
        idf = utils.get_idf(len(self.doc_info), self.tf_inverted_index)

        for doc_id, doc_length in self.doc_info.items():
            for term, qf in self.query_term_frequency.items():
                if self.tf_inverted_index.get(term) is None:
                    continue
                fi = self.tf_inverted_index.get(term).get(doc_id)
                if fi is None:
                    continue
                score = (self.k1 + 1) * qf * fi * (self.k3 + 1) * idf.get(term) / \
                    ((fi + self.k1 * (1 - self.b + self.b * (doc_length / avg_dl))) * (self.k3 + qf))
                self.bm25plus_score[doc_id] += score

        # 去掉得分为0的文档
        bm25plus_score = {key: value for key, value in self.bm25plus_score.items() if value != 0.0}
        self.ranked_doc_ids = [k for k, v in sorted(bm25plus_score.items(), key=lambda x: x[1], reverse=True)]
        if len(self.ranked_doc_ids) > Constants.NUM_OF_RET_TOP_DOC:
            for i in range(Constants.NUM_OF_RET_TOP_DOC, len(self.ranked_doc_ids)):
                if bm25plus_score[self.ranked_doc_ids[i]] < bm25plus_score[self.ranked_doc_ids[0]] / 2:
                    self.ranked_doc_ids = self.ranked_doc_ids[:i]
                    break

        utils.debug_print(bm25plus_score)
        end_time = time.time()
        utils.debug_print('Time for execute_query: ' + str(end_time - start_time) + 'seconds')


if __name__ == '__main__':
    import os
    from doc_parsers.pdf_parser import Pypdf2Parser
    from index_managers.tf_inverted_indexer import TfInvertedIndexer
    from query_managers.query_manager import QueryManager

    # 步骤一：解析原始文档
    my_start_time = time.time()
    pdf_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    my_pdf_parser = Pypdf2Parser(pdf_dir)
    my_pdf_parser.parse_docs()
    print('Time for parsing docs: ' + str(time.time() - my_start_time) + ' seconds')

    # 步骤二：构建索引
    my_start_time = time.time()
    my_doc_ids = list(my_pdf_parser.doc_info.keys())
    my_vocabulary = my_pdf_parser.vocabulary
    my_term_doc_tf_matrix = my_pdf_parser.term_doc_tf_matrix
    my_inverted_indexer = TfInvertedIndexer(my_doc_ids, my_vocabulary, my_term_doc_tf_matrix)
    my_inverted_indexer.build_index()
    print('Time for building index: ' + str(time.time() - my_start_time) + ' seconds')

    while True:
        # 步骤三：用户输入查询并解析查询（自由文本，且其预处理逻辑应与原始文档解析中的逻辑保持一致）
        my_query_str = input('请输入查询：')
        if my_query_str == 'exit':
            break

        my_query_manager = QueryManager(my_query_str)
        my_query_manager.parse_query()
        print(my_query_manager.query_term_list)

        # 步骤四：依据用户查询在现有索引上进行基于BM25+模型的文档搜索及排序
        my_start_time = time.time()
        my_query_term_frequency = my_query_manager.query_term_frequency
        my_index = my_inverted_indexer.index
        my_doc_info = my_pdf_parser.doc_info
        my_bm25plus_query_model = BM25PLUSQueryModel(my_query_term_frequency, my_index, my_doc_info)
        my_bm25plus_query_model.execute_query()
        my_ranked_doc_ids = my_bm25plus_query_model.ranked_doc_ids

        if len(my_ranked_doc_ids) == 0:
            print("未查询到任何文档")
        else:
            print('相关文档如下所示：')
            for my_doc_index in range(len(my_ranked_doc_ids)):
                print(my_ranked_doc_ids[my_doc_index])

        print('Time for executing query: ' + str(time.time() - my_start_time) + ' seconds')

    print('查询结束，退出系统')
