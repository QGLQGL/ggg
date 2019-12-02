# %%

import pkuseg
seg = pkuseg.pkuseg()

if __name__ == '__main__':
    print(seg.cut('北京大学语言计算机与机器学习研究组研制推出了一套全新的中文分词工具包，它简单易用，支持多领域分词，等。'))

# %%
def is_chinese(uchar):
    if uchar >= u'\\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False



# %%
def format_str(content):
    content_str = ''
    for i in content:
        if is_chinese(i):
            content_str = content_str + i
    return content_str

# %%
filenames = ['a.txt', 'b.txt', 'c.txt']

if __name__ == '__main__':
    corpus = []
    for name in filenames:
        with open(name,'r') as f:
            str = f.read()
            str = format_str(str)
            str = seg.cut(str)
            corpus.append(' '.join(str))

# %%
