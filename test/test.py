import time
import os
from itertools import zip_longest
from utilities import Constants
from gensim.models import KeyedVectors
from doc_parsers.pdf_parser import Pypdf2Parser
from index_managers.tf_inverted_indexer import TfInvertedIndexer
from query_managers.query_manager import QueryManager
from query_models.bm25plus_query_model import BM25PLUSQueryModel
from query_models.bm25_word2vec_query_model import WORD2VECQueryModel

if __name__ == '__main__':
    query_file = r'.\query.txt'
    result_file = r'.\result.txt'

    # 加载模型
    my_start_time = time.time()
    model_file = os.path.join('..', Constants.WORD2VEC_MODEL_FILE)
    print('Now loading model. This may take 2 to 3 minutes...')
    my_word2vec_model = KeyedVectors.load_word2vec_format(model_file, binary=False)
    print('Time for loading model: ' + str(time.time() - my_start_time) + ' seconds')

    # 步骤一：解析原始文档
    my_start_time = time.time()
    pdf_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    my_pdf_parser = Pypdf2Parser(pdf_dir)
    my_pdf_parser.parse_docs()
    print('Time for parsing docs: ' + str(time.time() - my_start_time) + ' seconds')
    my_doc_info = my_pdf_parser.doc_info

    # 步骤二：构建索引
    my_start_time = time.time()
    my_doc_ids = list(my_pdf_parser.doc_info.keys())
    my_vocabulary = my_pdf_parser.vocabulary
    my_term_doc_tf_matrix = my_pdf_parser.term_doc_tf_matrix
    my_inverted_indexer = TfInvertedIndexer(my_doc_ids, my_vocabulary, my_term_doc_tf_matrix)
    my_inverted_indexer.build_index()
    print('Time for building index: ' + str(time.time() - my_start_time) + ' seconds')
    my_index = my_inverted_indexer.index

    with open(result_file, 'w', encoding='utf-8') as out_f:
        # out_f.write('BM25PLUS_model\t\t\t\t\t\t\tword2vec_model\t\t\t\t\t\t\tcombined_result\n')
        out_f.write(f"{'BM25PLUS_model':<40} {'word2vec_model':<40} {'combined_result':<40}\n")
        # 步骤三：用户输入查询并解析查询
        with open(query_file, 'r', encoding='utf-8') as in_f:
            for query_number, query in enumerate(in_f):
                query = query.strip()
                if query == 'exit':
                    break
                my_query_manager = QueryManager(query, my_word2vec_model)
                my_query_manager.parse_query()
                print(my_query_manager.query_term_list)

                # 步骤四：依据用户查询进行文档搜索及排序
                # BM25PLUS模型
                my_query_term_frequency = my_query_manager.query_term_frequency
                my_query_model_0 = BM25PLUSQueryModel(my_query_term_frequency, my_index, my_doc_info)
                my_query_model_0.execute_query()
                my_ranked_doc_ids_0 = my_query_model_0.ranked_doc_ids

                # WORD2VEC模型
                my_query_term_relevance = my_query_manager.query_term_relevance
                my_doc_seg_dict = my_pdf_parser.doc_seg_dict
                my_query_model_1 = WORD2VECQueryModel(my_query_term_relevance, my_doc_seg_dict,
                                                      my_doc_info, my_word2vec_model)
                my_query_model_1.execute_query()
                my_ranked_doc_ids_1 = my_query_model_1.ranked_doc_ids

                # 综合加权计算结果
                score_0_dict = my_query_model_0.bm25plus_score
                score_1_dict = my_query_model_1.word2vec_score
                combined_score = dict().fromkeys(score_0_dict.keys(), 0.0)
                # 加权处理，BM25+分数低很多，给较大权重。且该分数是文档实际包含该词项时的分数，也应当具有较大权重
                for key in combined_score.keys():
                    combined_score[key] = score_1_dict[key]
                    if score_0_dict[key] > 0.0:
                        combined_score[key] += score_0_dict[key] * 100

                # 去掉得分为0的文档
                combined_score = {key: value for key, value in combined_score.items() if value != 0.0}
                combined_ranked_doc_ids = [k for k, v in sorted(combined_score.items(),
                                                                key=lambda x: x[1], reverse=True)]
                for i in range(len(combined_ranked_doc_ids)):
                    if combined_score[combined_ranked_doc_ids[i]] < combined_score[combined_ranked_doc_ids[0]] / 2:
                        combined_ranked_doc_ids = combined_ranked_doc_ids[:i]
                        break

                # 写入结果文件
                out_f.write(str(query_number + 1) + ': ' + str(query) + ':\n')
                merged_doc_ids = list(zip_longest(my_ranked_doc_ids_0, my_ranked_doc_ids_1, combined_ranked_doc_ids,
                                                  fillvalue='None'))
                for items in merged_doc_ids:
                    # out_f.write(str(items[0]) + ' \t' + str(items[1]) + ' \t' + str(items[2]) + '\n')
                    out_f.write(f"{items[0]:<40} {items[1]:<40} {items[2]:<40}\n")
                out_f.write('\n')

    print('测试结束，退出系统')
