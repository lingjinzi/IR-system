from gensim.models import Word2Vec
from gensim.corpora import WikiCorpus
from gensim.models.word2vec import LineSentence
from opencc import OpenCC
import jieba
from utilities import Constants
from utilities import utils

if __name__ == '__main__':
    # xml to txt
    def xml2txt_generator(input_xml_file):
        wiki_corpus = WikiCorpus(input_xml_file, dictionary=[])
        index = 0
        for text in wiki_corpus.get_texts():
            index += 1
            if index % 10000 == 0:
                print("Processed " + str(index) + " articles")
            yield ' '.join(text)

    # 繁化简
    def t2s_generator(text_generator):
        for text in text_generator:
            yield opencc.convert(text)

    # 分词
    def seg_text_generator(simplified_text_generator):
        for text in simplified_text_generator:
            words = jieba.cut(text)
            for word in words:
                # 去掉非中文
                if utils.is_all_chinese(word):
                    # 去掉停用词
                    if word not in Constants.STOP_WORDS:
                        yield word

    opencc = OpenCC('t2s')
    xml_file_path = r'.\corpus\zhwiki-20240501-pages-articles-multistream.xml.bz2'
    text_file_path = r'.\corpus\zh_wiki.txt'
    model_path = r'.\text.vector'

    print('xml to txt...')
    with open(text_file_path, 'w', encoding='utf-8') as out_f:
        # xml to text
        text_gen = xml2txt_generator(xml_file_path)

        # 繁化简
        simplified_text_gen = t2s_generator(text_gen)

        # 分词并写入txt文件
        for seg_text in seg_text_generator(simplified_text_gen):
            out_f.write(seg_text + '\n')

    print('txt to model...')
    model = Word2Vec(LineSentence(text_file_path), vector_size=100, window=5, min_count=5, workers=4)
    model.wv.save_word2vec_format(model_path)
    print('model saved.')







