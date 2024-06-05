import statistics
import time
from utilities import utils
from utilities import Constants


class WORD2VECQueryModel:
    def __init__(self, query_term_relevance, doc_seg_dict, doc_info, word2vec_model, b=0.5, k1=1.2, k3=10):
        self.query_term_relevance = query_term_relevance
        self.doc_info = doc_info
        self.doc_seg_dict = doc_seg_dict
        self.model = word2vec_model
        self.word2vec_score = dict.fromkeys(self.doc_info.keys(), 0.0)
        self.ranked_doc_ids = list()

        # BM25+算法参数
        self.b = b
        self.k1 = k1
        self.k3 = k3

    def execute_query(self):
        start_time = time.time()

        # 参数计算
        avg_dl = statistics.mean(self.doc_info.values())

        for doc_id, doc_length in self.doc_info.items():
            for term, qr in self.query_term_relevance.items():
                ri = utils.word_document_similarity(term, self.doc_seg_dict[doc_id], self.model)
                score = (self.k1 + 1) * qr * ri * (self.k3 + 1) / \
                        ((ri + self.k1 * (1 - self.b + self.b * (doc_length / avg_dl))) * (self.k3 + qr))
                self.word2vec_score[doc_id] += score

        # 去掉得分为0的文档
        word2vec_score = {key: value for key, value in self.word2vec_score.items() if value != 0.0}
        self.ranked_doc_ids = [k for k, v in sorted(word2vec_score.items(), key=lambda x: x[1], reverse=True)]
        for i in range(len(self.ranked_doc_ids)):
            if word2vec_score[self.ranked_doc_ids[i]] < word2vec_score[self.ranked_doc_ids[0]] / 2:
                self.ranked_doc_ids = self.ranked_doc_ids[:i]
                break

        utils.debug_print(word2vec_score)
        end_time = time.time()
        utils.debug_print('Time for execute_query: ' + str(end_time - start_time) + 'seconds')


if __name__ == '__main__':
    import os
    from doc_parsers.pdf_parser import Pypdf2Parser
    from index_managers.tf_inverted_indexer import TfInvertedIndexer
    from query_managers.query_manager import QueryManager
    from gensim.models import KeyedVectors

    # 加载模型
    my_start_time = time.time()
    model_file = Constants.WORD2VEC_MODEL_FILE
    print('Now loading model. This may take 2 to 3 minutes...')
    my_word2vec_model = KeyedVectors.load_word2vec_format(model_file, binary=False)
    print('Time for loading model: ' + str(time.time() - my_start_time) + ' seconds')

    # 步骤一：解析原始文档
    my_start_time = time.time()
    pdf_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    my_pdf_parser = Pypdf2Parser(pdf_dir)
    my_pdf_parser.parse_docs()
    print('Time for parsing docs: ' + str(time.time() - my_start_time) + ' seconds')

    # 步骤二：构建索引
    # my_start_time = time.time()
    # my_doc_ids = list(my_pdf_parser.doc_info.keys())
    # my_vocabulary = my_pdf_parser.vocabulary
    # my_term_doc_tf_matrix = my_pdf_parser.term_doc_tf_matrix
    # my_inverted_indexer = TfInvertedIndexer(my_doc_ids, my_vocabulary, my_term_doc_tf_matrix)
    # my_inverted_indexer.build_index()
    # print('Time for building index: ' + str(time.time() - my_start_time) + " seconds")

    while True:
        # 步骤三：用户输入查询并解析查询（自由文本，且其预处理逻辑应与原始文档解析中的逻辑保持一致）
        my_query_str = input('请输入查询：')
        if my_query_str == 'exit':
            break

        my_query_manager = QueryManager(my_query_str, my_word2vec_model)
        my_query_manager.parse_query()
        print(my_query_manager.query_term_list)

        # 步骤四：依据用户查询基于语义模型进行文档搜索即排序
        my_start_time = time.time()
        my_query_term_relevance = my_query_manager.query_term_relevance
        my_doc_info = my_pdf_parser.doc_info
        my_doc_seg_dict = my_pdf_parser.doc_seg_dict
        my_word2vec_query_model = WORD2VECQueryModel(my_query_term_relevance, my_doc_seg_dict,
                                                     my_doc_info, my_word2vec_model)
        my_word2vec_query_model.execute_query()
        my_ranked_doc_ids = my_word2vec_query_model.ranked_doc_ids

        if len(my_ranked_doc_ids) == 0:
            print("未查询到任何文档")
        else:
            print('相关文档如下所示：')
            for my_doc_index in range(len(my_ranked_doc_ids)):
                print(my_ranked_doc_ids[my_doc_index])

        print('Time for executing query: ' + str(time.time() - my_start_time) + ' seconds')

    print('查询结束，退出系统')
