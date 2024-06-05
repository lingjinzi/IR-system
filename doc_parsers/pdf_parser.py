from PyPDF2 import PdfReader
import os
import time
from utilities import utils
from utilities import Constants
import jieba
import numpy as np
from opencc import OpenCC


class Pypdf2Parser:
    def __init__(self, parent_dir):
        self.parent_dir = parent_dir
        self.vocabulary = list()
        self.doc_seg_dict = None                    # 不去重
        self.doc_info = None                        # 文档信息(ids, length)
        self.term_doc_tf_matrix = None              # tf(term-frequency)表

    def parse_docs(self):
        start_time = time.time()
        doc_ids = utils.get_file_url_list(self.parent_dir)
        all_doc_seg_list = list()                   # 每个文档所有词，不去重
        doc_lengths = list()

        # 繁化简转化器
        converter = OpenCC('t2s')
        for file_index in range(len(doc_ids)):
            reader = PdfReader(doc_ids[file_index])

            # Print the number of pages in the PDF
            utils.debug_print(f"There are {len(reader.pages)} Pages")

            curr_doc_seg_list = list()
            # Go through every page and get the text
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                page_str = page.extract_text()
                # print(page_str)
                raw_seg_list = jieba.lcut(page_str)

                utils.debug_print(
                    'Start to process page ' + str(page_num + 1) + ' out of ' + str(len(reader.pages)))
                utils.debug_print('Before: ' + str(len(raw_seg_list)))
                # print("Paddle Mode: " + ','.join(raw_seg_list))

                seg_list = list()
                for token_index in range(len(raw_seg_list)):
                    # 去掉非中文词
                    if utils.is_all_chinese(raw_seg_list[token_index]):
                        # 繁化简
                        raw_seg_list[token_index] = converter.convert(raw_seg_list[token_index])
                        # 去掉停用词
                        if raw_seg_list[token_index] not in Constants.STOP_WORDS:
                            seg_list.append(raw_seg_list[token_index])

                utils.debug_print('After: ' + str(len(seg_list)))
                # print("Paddle Mode: " + ','.join(seg_list))

                self.vocabulary.extend(seg_list)
                self.vocabulary = list(set(self.vocabulary))
                utils.debug_print('Size of vocabulary: ' + str(len(self.vocabulary)))
                utils.debug_print(self.vocabulary)

                curr_doc_seg_list.extend(seg_list)

            # 不去重
            # curr_doc_seg_list = list(set(curr_doc_seg_list))
            doc_lengths.append(len(curr_doc_seg_list))
            all_doc_seg_list.append(curr_doc_seg_list)

        self.doc_seg_dict = {doc_id: seg_list for doc_id, seg_list in zip(doc_ids, all_doc_seg_list)}
        self.doc_info = {doc_id: length for doc_id, length in zip(doc_ids, doc_lengths)}

        avg_dl = np.mean(np.array(doc_lengths))

        # 构造矩阵
        shape = [len(doc_ids), len(self.vocabulary)]
        utils.debug_print('Shape of term_doc_tf_matrix: ' + str(shape))
        self.term_doc_tf_matrix = np.zeros(shape, dtype=float)

        for doc_index in range(len(doc_ids)):
            for term_index in range(len(all_doc_seg_list[doc_index])):
                curr_term = all_doc_seg_list[doc_index][term_index]
                curr_term_index = self.vocabulary.index(curr_term)
                self.term_doc_tf_matrix[doc_index, curr_term_index] += 1 / doc_lengths[doc_index]

        utils.debug_print('term_doc_tf_matrix: ')
        utils.debug_print(self.term_doc_tf_matrix)
        sparsity = np.count_nonzero(self.term_doc_tf_matrix) / (shape[0] * shape[1])
        utils.debug_print('Sparsity of term_doc_tf_matrix: ' + str(round(sparsity, 3)))

        end_time = time.time()
        utils.debug_print('Time for parse_docs: ' + str(end_time - start_time) + ' seconds')


if __name__ == '__main__':
    # 步骤一：解析原始文档，训练模型
    my_start_time = time.time()
    pdf_dir = os.path.abspath(os.path.join(os.getcwd(), "../archives"))
    my_pdf_parser = Pypdf2Parser(pdf_dir)
    my_pdf_parser.parse_docs()
    print('Time for parsing docs: ' + str(time.time() - my_start_time) + ' seconds')
    print(my_pdf_parser.doc_info)


