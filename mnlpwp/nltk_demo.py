import re
import string

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import TreebankWordTokenizer, WordPunctTokenizer, WhitespaceTokenizer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize.util import spans_to_relative, string_span_tokenize
from nltk.corpus import stopwords, gutenberg, alpino, webtext
from nltk.probability import FreqDist
from nltk.metrics import edit_distance, jaccard_distance, binary_distance, masi_distance
from nltk.metrics import BigramAssocMeasures
from nltk.util import ngrams
from nltk.collocations import BigramCollocationFinder, QuadgramCollocationFinder

import matplotlib
import matplotlib.pyplot as plt

from NLP.replacers import RegexpReplacer, WordReplacer


def test_sent_tokenize():
    text = 'Welcome readers. I hope you find it interesting. Please do reply.'
    tokens = sent_tokenize(text)  # 文本切分成句子
    print(tokens)


def test_word_tokenize():
    sentence = 'I hope you find it interesting.'
    sentence = "Don't hesitate to Ask questions."
    tokens = nltk.word_tokenize(sentence)  # 文本切分成单词  nltk.word_tokenize 就是 nltk.tokenize的word_tokenize
    print(tokens)
    tokens = word_tokenize(sentence)
    print(tokens)
    tokenizer = TreebankWordTokenizer()
    sentence = 'Have a nice day. I hope you find the book interesting.'
    tokens = tokenizer.tokenize(sentence)
    print(tokens)


def test_tokenizer():
    text = "Don't hesitate to ask questions."
    tokenizer = TreebankWordTokenizer()
    print(tokenizer.tokenize(text))
    tokenizer = WordPunctTokenizer()
    print(tokenizer.tokenize(text))
    tokenizer = WhitespaceTokenizer()
    print(tokenizer.tokenize(text))

    print(nltk.word_tokenize(text))


def test_regexp_tokenizer():
    text = "Don't hesitate to Ask questions."
    tokenizer = RegexpTokenizer(r"[\w']+")
    print(tokenizer.tokenize(text))
    print(nltk.regexp_tokenize(text, r'[\w]+'))
    tokenizer = RegexpTokenizer(r"[A-Z]\w+")  # 大写开头的词
    print(tokenizer.tokenize(text))


def test_span_tokenize():
    text = "She secured 90.56 % in class X \n. She is a meritorious student\n"
    tokenizer = WhitespaceTokenizer()
    print(tokenizer.tokenize(text))
    print(list(tokenizer.span_tokenize(text)))  # 输出每个切分出来的词的位置[start_index, end_index)左闭右开
    print(list(spans_to_relative(tokenizer.span_tokenize(text))))  # 输出每个词的跨度
    print(list(string_span_tokenize(text, ' ')))  # 指定分隔符，输出每个切分出来的词的位置[start_index, end_index)左闭右开


def test_remove_punctuation():
    text = [" It is a pleasant evening.", "Guests, who came from US arrived at the venue", "Food was tasty."]
    token_sentence_list = [nltk.word_tokenize(sentence) for sentence in text]
    print(re.escape(string.punctuation))
    pattern = re.compile('[%s]' % re.escape(string.punctuation))
    result_no_punctuation = []
    for token_sentence in token_sentence_list:
        no_punctuation = []
        for token in token_sentence:
            temp = pattern.sub('', token)
            if temp:
                no_punctuation.append(temp)
        result_no_punctuation.append(no_punctuation)
    print(result_no_punctuation)


def test_stopwords():
    words = ['Guests', 'who', 'came', 'from', 'US', 'arrived', 'at', 'the', 'venue']
    stops = set(stopwords.words('english'))
    print(len(stops))
    print(words)
    print([word for word in words if word not in stops])  # 去除停止词
    print(stopwords.fileids())  # 查看支持的语言
    print(*stopwords.words('chinese'), sep='\n')  # 列出中文停止词


def test_replacer():
    text = "Don't hesitate to Ask questions."
    replacer = RegexpReplacer()
    print(text)
    print(replacer.replace(text))
    words = word_tokenize(text)
    print(words)
    words = word_tokenize(replacer.replace(text))
    print(words)
    stops = stopwords.words('english')
    print([word for word in words if word not in stops])


def test_repeat():
    words = ['lotttt', 'ohhhhh', 'oooohhhh', 'happy']
    replacer = RegexpReplacer()
    print([replacer.replace_repeat(word) for word in words])


def test_word_replacer():
    replacer = WordReplacer({'congrats': 'congratulations'})
    print(replacer.replace('congrats'))
    print(replacer.replace('maths'))


def test_zipf():
    matplotlib.use('TkAgg')
    fd = FreqDist()
    for text in gutenberg.fileids():
        for word in gutenberg.words(text):
            fd[word.lower()] += 1
    ranks = []
    freqs = []
    for rank, word in enumerate(fd):
        ranks.append(rank + 1)
        freqs.append(fd[word])
    plt.loglog(ranks, freqs)
    plt.xlabel('frequency(f)', fontsize=14, fontweight='bold')
    plt.ylabel('rank(r)', fontsize=14, fontweight='bold')
    plt.grid(True)
    plt.show()


def test_similarity():
    word1 = 'relate'
    word2 = 'relation'
    dis = edit_distance(word1, word2)  # 编辑距离
    print(f'{word1} and {word2} the distance is {dis}')
    x = {10, 20, 30, 40}
    y = {20, 30, 60}
    print(jaccard_distance(x, y))  # Jaccard系数
    print(binary_distance(x, y))  # 二进制距离
    print(masi_distance(x, y))  # Masi距离


def test_ngram():
    # unigrams = ngrams(alpino.words(), 1)
    # for i in unigrams:
    #     print(i)

    words = [t.lower() for t in webtext.words('grail.txt')]
    # print(tokens)
    stops = set(stopwords.words('english'))
    finder = BigramCollocationFinder.from_words(words)
    finder.apply_word_filter(lambda w: len(w) < 3 or w in stops)
    print(finder.nbest(BigramAssocMeasures.likelihood_ratio, 10))

    text = 'Hardwork is the key to success. Never give up!'
    words = nltk.wordpunct_tokenize(text)
    finder = BigramCollocationFinder.from_words(words)
    bigram_measures = BigramAssocMeasures()
    value = finder.score_ngrams(bigram_measures.raw_freq)
    print(sorted((bigram, score) for bigram, score in value))

    # bigram_tokens = ngrams(alpino.words(), 2)
    # for i in bigram_tokens:
    #     print(i)

    # trigram_tokens = ngrams(alpino.words(), 3)
    # for i in trigram_tokens:
    #     print(i)

    fourgrams = QuadgramCollocationFinder.from_words(words)
    for fourgram, freq in fourgrams.ngram_fd.items():
        print(fourgram, freq)


def main():
    # test_sent_tokenize()
    # test_word_tokenize()
    # test_tokenizer()
    # test_regexp_tokenizer()
    # test_span_tokenize()
    # test_remove_punctuation()
    # test_stopwords()
    # test_replacer()
    # test_repeat()
    # test_word_replacer()
    # test_zipf()
    # test_similarity()
    test_ngram()


if __name__ == '__main__':
    main()
