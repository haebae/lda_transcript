#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv('test_0.csv')


# In[3]:


data


# In[4]:


import itertools
noun = list(itertools.chain(data['token_noun']))   #리스트 언패킹
print(type(noun))


# In[5]:


print(type(noun[0]))


# In[6]:


import ast
L = []
for i in range(len(noun)) : 
    a = ast.literal_eval(noun[i])
    for j in range(len(a)) : 
        L.append(a[j])


# In[7]:


type(L)


# In[8]:


# strL = ",".join(L)
# print(type(strL))


# In[11]:


#명사만 추출한 리스트를 바탕으로 단어 빈도별 목록을 생성.
from gensim import corpora, models
noun_dic=corpora.Dictionary([L]) #딕셔너리 클래스로 사전생성, 각 단어별 id도 함께 생성
noun_dic.token2id                               #각 단어 별 생성된 id 확인 .. 리스트에 리스트형식


# In[12]:


from collections import Counter #단어들을 쉽게 집계하기 위해서 사용
count = Counter(L) #리스트 원소의 개수가 계산됨
top = dict(count.most_common(50)) # 상위 50개 출력하기
top


# In[13]:


corpus = [noun_dic.doc2bow(text) for text in [L]]  
#문서의 단어들을 단어의 id와 빈도수로 수치화
corpus 


# In[14]:


#토픽 개수에 따라 분석의 혼잡도와 일관성 분석 후, 최선의 토픽 개수를 정하여 토픽 모델링.
import gensim
from gensim.models import CoherenceModel

Lda = gensim.models.ldamodel.LdaModel  #LDA기법: 확률을 바탕으로 단어가 특정 주제에 존재할 확률과 문서에 특정 주제가 존재할 확률을 결합확률로 추정하여 토픽추출
perplexity_score=[]
coherence_score=[]

for i in range(1,10): #토픽 개수가 1,2,3,4,5,6,7,8,9인 9가지 경우 각각의 혼잡도와 일관성을 측정
    ldamodel=Lda(corpus, num_topics=i, id2word=noun_dic, passes=15, random_state=0)  #passes: 모델 학습시 전체 코퍼스에서 모델을 학습시키는 빈도 #LDA는 확률적 알고리즘이기때문에 random_state 를 바꾸면 결과가 바뀌니 주의하자.
    perplexity_score.append(ldamodel.log_perplexity(corpus)) #혼잡도
    coherence_score.append(CoherenceModel(model=ldamodel, texts=[L], dictionary=noun_dic, coherence='c_v').get_coherence()) #일관성
    print(i, 'process complete')


# In[15]:


plt.plot(range(1,10),perplexity_score,'r',marker='^') #(x,y,color)
plt.xlabel("number of topics")
plt.ylabel("perplexity score") #혼잡성
plt.show()


# In[16]:


plt.plot(range(1,10),coherence_score,'b',marker='o')
plt.xlabel("number of topics")
plt.ylabel("coherence score") #일관성
plt.show()


# In[17]:


#최적의 토픽수로 토픽 모델링 진행->topic 3,6,7개로 확정.
# noun_lda=Lda(corpus, num_topics=7, id2word=noun_dic, passes=15, random_state=0)
noun_lda=Lda(corpus, num_topics=7, id2word=noun_dic)
topics=noun_lda.print_topics(num_words=20) #토픽별 5 단어씩 출력
for topic in topics: #num_topics=3이므로 3개로 압축된 토픽을 각각 출력.
    print(topic)


# In[18]:


import pyLDAvis.gensim_models
vis=pyLDAvis.gensim_models.prepare(noun_lda, corpus, noun_dic)
pyLDAvis.display(vis)


    