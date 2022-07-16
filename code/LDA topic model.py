from gensim.models.tfidfmodel import TfidfModel
from gensim import corpora, models

file_name = 'data/covid_rumour_tokens.txt'
with open(file_name) as f:
    all_line = f.readlines()
for i,sent in enumerate(all_line):
    # print(sent)
    temp = sent.strip()
    # print(temp)
    temp = temp.split()
    # print(temp)
    # input()
    all_line[i] = temp
# print(all_line[:5])
# input()

dictionary1 = corpora.Dictionary(all_line)

corpus1 = [dictionary1.doc2bow(text) for text in all_line]
print(corpus1)

tf_idf_model = TfidfModel(corpus1, normalize=False)
word_tf_tdf = list(tf_idf_model[corpus1])
print(word_tf_tdf)

print(len(corpus1))
print(len(word_tf_tdf))

lda1 = models.ldamodel.LdaModel(corpus=word_tf_tdf, id2word=dictionary1, num_topics=10, update_every=1, chunksize =10000, passes=1)
print(lda1.print_topics(num_topics=20, num_words=10))