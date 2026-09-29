

import numpy as np
#人为定义词的二维向量
embeddings = {
    "king": np.array([0.9,0.8]),
    "queen": np.array([0.9,0.2]),
    "man": np.array([0.7,0.9]),
    "woman": np.array([0.7,0.3]),
    "mteacher": np.array([0.4,0.7]),
    "wteacher": np.array([0.4,0.1])
}



#定义余弦相似度函数
def cosine_similarity(vector1,vector2):
    dot_product = np.dot(vector1,vector2)#向量积
    norm_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)#模的积
    return dot_product / norm_product #返回两者相比的结果，也就是两个向量夹角的余弦值

#king-man+woman
result_vector = embeddings["king"]-embeddings["man"]+embeddings["woman"]

woman_teacher_vector = embeddings["mteacher"]-embeddings["man"]+embeddings["woman"]


#比较result_vector和queen的相似度
CosSim = cosine_similarity(result_vector,embeddings["queen"])
CosSim1 = cosine_similarity(woman_teacher_vector,embeddings["wteacher"])

print(f"king - man + woman 的结果向量: {result_vector}")
print(f"该结果与 'queen' 的相似度: {CosSim:.4f}")

print(f"女老师 的结果向量: {woman_teacher_vector}")
print(f"该结果与 '女老师' 的相似度: {CosSim1:.4f}")