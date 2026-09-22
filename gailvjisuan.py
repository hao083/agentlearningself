'''
LLM出现之前，统计模型广泛应用于文本生成，条件概率的计算就是统计模型中很重要的一部分
模拟计算 
"datawhale agent learns datawhale agent works“中生成datawhale agent learns的概率
'''

import collections
store = "datawhale agent learns datawhale agent works"#初始语料库
tokens=store.split()#按照空格进行初始语料库分割，效果为分割为独立的单词
cnt_tokens=len(tokens)#计算总数

#首先计算P(datawhale)
cnt_datawhale = tokens.count("datawhale")
p_datawhale = cnt_datawhale / cnt_tokens
print(f"First: P(datawhale) = {cnt_datawhale} / {cnt_tokens} = {p_datawhale:.3f}")
print("\n")
#接着计算P（agent|datawhale）
bigram = zip(tokens,tokens[1:])
bigram_cnt = collections.Counter(bigram)
cnt_datawhale_agent = bigram_cnt[('datawhale','agent')]#计算datawhale agent连续出现的次数
p_agent_givenby_datawhale = cnt_datawhale_agent / cnt_datawhale
print(f"Second: P(agent_givenby_datawhale) = \
{cnt_datawhale_agent} / {cnt_datawhale} = \
{p_agent_givenby_datawhale:.3f}")
print("\n")
#再计算P（learns|agent）
cnt_agent_learns = bigram_cnt[('agent', 'learns')]
cnt_agent = tokens.count('agent')
p_learns_given_agent = cnt_agent_learns / cnt_agent
print(f"Third: P(learns|agent) = \
{cnt_agent_learns}/{cnt_agent} = \
{p_learns_given_agent:.3f}")
print("\n")
#最后将上述概率连乘可得最终的生成概率
p_sentence = p_datawhale * p_agent_givenby_datawhale * p_learns_given_agent
print(f"At last: P('datawhale agent learns') ≈ {p_datawhale:.3f} *\
{p_agent_givenby_datawhale:.3f} * {p_learns_given_agent:.3f} =\
{p_sentence:.3f}")

