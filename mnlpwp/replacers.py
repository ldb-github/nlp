import re
from nltk.corpus import wordnet


# 缩略词替换正则
replacement_patterns = [
    (r"won't", "will not"),
    (r"can't", "can not"),
    (r"i'm", "i am"),
    (r"ain't", "is not"),
    (r"(\w+)'ll", r"\g<1> will"),
    (r"(\w+)n't", r"\g<1> not"),
    (r"(\w+)'ve", r"\g<1> have"),
    (r"(\w+)'s", r"\g<1> is"),
    (r"(\w+)'re", r"\g<1> are"),
    (r"(\w+)'d", r"\g<1> would")
]


class RegexpReplacer:
    def __init__(self, patterns=None):
        if patterns is None:
            patterns = replacement_patterns
        self.patterns = [(re.compile(regex), repl) for regex, repl in patterns]
        self.repeat_regexp = re.compile(r"(\w*)(\w)\2(\w*)")
        self.repeat_repl = r'\1\2\3'

    def replace(self, text):
        s = text
        for pattern, repl in self.patterns:
            s, count = re.subn(pattern, repl, s)
        return s

    def replace_repeat(self, word):
        if wordnet.synsets(word):
            return word
        repl_word = self.repeat_regexp.sub(self.repeat_repl, word)
        if repl_word != word:
            return self.replace_repeat(repl_word)
        else:
            return repl_word


class WordReplacer:
    def __init__(self, word_map):
        self.word_map = word_map

    def replace(self, word):
        return self.word_map.get(word, word)



