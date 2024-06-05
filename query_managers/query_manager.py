import jieba
import time
from collections import Counter
from utilities import utils
from utilities import Constants


class QueryManager:
    def __init__(self, query_str, model=None):
        self.query_str = query_str
        self.word_vec_model = model
        self.query_term_list = None
        self.query_term_relevance = None
        self.query_term_frequency = None

    def parse_query(self):
        start_time = time.time()
        self.query_term_list = jieba.lcut(self.query_str)

        # 只保留全中文词
        self.query_term_list = [term for term in self.query_term_list if utils.is_all_chinese(term)]

        # 去掉停用词
        self.query_term_list = utils.remove_stop_words(self.query_term_list)

        # 计算词项在查询语句中的频率
        self.query_term_frequency = Counter(self.query_term_list)

        # 计算词项在查询语句中的代表性
        if self.word_vec_model:
            query_term_set = set(self.query_term_list)
            self.query_term_relevance = dict().fromkeys(query_term_set, 0.0)
            for word in query_term_set:
                self.query_term_relevance[word] = (utils.word_document_similarity(
                    word, self.query_term_list, self.word_vec_model))

        end_time = time.time()
        utils.debug_print('Time for parse_query: ' + str(end_time - start_time) + ' seconds')


if __name__ == '__main__':
    import os
    from doc_parsers.pdf_parser import Pypdf2Parser
    from index_managers.tf_inverted_indexer import TfInvertedIndexer
    from utilities import utils
    from gensim.models import KeyedVectors

    # 加载模型
    my_start_time = time.time()
    model_file = Constants.WORD2VEC_MODEL_FILE
    my_word2vec_model = KeyedVectors.load_word2vec_format(model_file, binary=False)
    print('Time for loading model: ' + str(time.time() - my_start_time) + ' seconds')

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

        my_query_manager = QueryManager(my_query_str, my_word2vec_model)
        my_query_manager.parse_query()
        print(my_query_manager.query_term_relevance)
        print(my_query_manager.query_term_frequency)
