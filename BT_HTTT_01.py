#!/usr/bin/env python
# coding: utf-8

# In[ ]:


BAI 1


# In[6]:


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
# # Read csv
# cols = ['user','item', 'rating']
# df = pd.read_csv('rating.csv',sep =',', names =cols, encoding ='latin1')
# print(df)
# Y = df.as_matrix()
# print('ma tran du lieu rating:\n',Y)
# n_users = np.max(A[:,0]) +1
# n_items = np.max(A[:,1]) +1
# print('so users:', n_users)
# print('so items:', n_items)
A = np.array([[1,4,5,0,3],
              [5,1,0,5,2],
              [4,1,2,5,0],
              [0,3,4,0,4]])
print('A', A)


# In[7]:


# Tính sim(): độ giống nhau giữa các user : (Joe, Ann, Mary, Steve)
Sim = np.zeros((4,4))
# Cách 1: dùng hàm cosine_similarity
Sim = cosine_similarity(A,A)
print('library sim user: \n', Sim)
# Cách 2: tính cosin
def tu(A,i,j):
    return  sum(A[i]*A[j])

def mau(A ,i,j):
#     print('A[i]', A[i])
#     print('A[j]:', A[j])
    a =0
    b = 0
    for k in range(0, A.shape[1]):
        if A[i][k]!=0 and A[j][k] !=0:
            a += A[i][k]**2
            b += A[j][k]**2
    return np.sqrt(a*b)
def cosin(A):
    for i in range(0, A.shape[0]):
        for j in range(0, A.shape[0]):
            if(i==j):
                Sim[i][j] ==1
            else:
                Sim[i][j] == tu(A,i,j)/mau(A,i,j)
    return Sim
print('similar user: \n', cosin(A))


# In[3]:


# A = np.array([[1,4,5,0,3],
#               [5,1,0,5,2],
#               [4,1,2,5,0],
#               [0,3,4,0,4]])
def predict(u,i, k=2):
    # predict rating of user u for item i ( k: chọn k user để đánh giá)
    # step 1: find all user who rated i
    user_rated_i = np.where(A[:,i] !=0)[0] # user rated i
    print(user_rated_i)
    # step 2: find similar of user current with user dif
    sim = Sim[u, user_rated_i]
    print(sim)
    # step 3: find k most user similar (sorted)
    a = (np.argsort(sim))[-k:] # lấy k chỉ số cuối
    nearest_s =sim[a] 
    print(nearest_s)
    print(a)
    # step 4: how each user 'nearest' rated item i
    rating = A[user_rated_i[a],i]
    print(rating)
    r_bar = (rating*nearest_s).sum()/(np.abs(nearest_s).sum()) # predict rating of user u for item i
    print('r_bar', r_bar)
    return np.round(r_bar)
# predict(1,2)
# predict(0,3)
# dự đoán đánh giá của user u
def print_recommend():
    B = A.copy() #matrix predict
    for u in range(0, A.shape[0]):
        for i in range(0, A.shape[1]):
            if A[u,i] ==0:
                # lấy vi tri chưa đánh giá
                id = i
                B[u,i] = predict(u,id)
    print('Ma tran truoc khi predict:\n', A)
    print('Ma tran rating sau khi predict:\n', B)
    
print_recommend()


# In[4]:


I = A.T # chuyển vị ma trận 
print(I)
# Tím SIM
Sim_item = cosine_similarity(I,I)
print(Sim_item)

# predict
def predict(u,i, k=2):
    # predict rating of user u for item i ( k: chọn k user để đánh giá)
    # step 1: find all user who rated i
    user_rated_i = np.where(I[:,i] !=0)[0] # user rated i
    print(user_rated_i)
    # step 2: find similar of user current with user dif
    sim = Sim_item[u, user_rated_i]
    print(sim)
    # step 3: find k most user similar (sorted)
    a = (np.argsort(sim))[-k:] # lấy k chỉ số cuối
    nearest_s =sim[a] 
    print(nearest_s)
    print(a)
    # step 4: how each user 'nearest' rated item i
    rating = I[user_rated_i[a],i]
    print(rating)
    r_bar = (rating*nearest_s).sum()/(np.abs(nearest_s).sum()) # predict rating of user u for item i
    print('r_bar', r_bar)
    return np.round(r_bar)
# predict(1,2)
# predict(0,3)
# dự đoán đánh giá của user u
def print_recommend():
    B = I.copy() #matrix predict
    for u in range(0, I.shape[0]):
        for i in range(0, I.shape[1]):
            if I[u,i] ==0:
                # lấy vi tri chưa đánh giá
                id = i
                B[u,i] = predict(u,id)
    print('Ma tran truoc khi predict:\n', I.T)
    print('Ma tran rating sau khi predict:\n', B.T)
    
print_recommend()


# In[ ]:


BAI 2


# In[5]:


A = np.array([[1,4,5,0,3],
              [5,1,0,5,2],
              [4,1,2,5,0],
              [0,3,4,0,4]])
print('A', A)
# Bước 1. Tính nguy
def nguy():
    M =[]
    for i in range(0, A.shape[0]):
        for j in range(0, A.shape[1]):
            if A[i,j] !=0:
                M.append(A[i,j])
    print(M)
    nguy = np.mean(M)
    print('nguy:', nguy)
    return nguy
nguy = nguy()
# Bước 2. Tính bu
def bu():
    tu =0 # tong cac( rating- nguy) trên tử
    num =0 
    arr = []
    for i in range(0, A.shape[0]):
        for j in range(0, A.shape[1]):
            if(A[i,j]!=0):
                tu += (A[i,j] - nguy)
                num +=1
        arr.append( tu/num)
    return arr
bu = bu()
print('bu', bu)
def bi():
    tu =0 # tong cac( rating- nguy) trên tử
    num =0 
    arr = []
    B = A.T
    for i in range(0, B.shape[0]):
        for j in range(0, B.shape[1]):
            if(B[i,j]!=0):
                tu += (B[i,j] - nguy)
                num +=1
        arr.append( tu/num)
    return arr
bi = bi()
print('bi', bi)
def rui(u, i):
    return nguy + bu[u] + bi[i]
print('rui(1,2)',rui(1,2))
def print_cold_start_problem():
    for i in range(0, A.shape[0]):
        for j in range(0, A.shape[1]):
            if A[i][j] ==0:
                A[i][j] = rui(i,j)
    return A
print('Matrix truoc khi du doan:\n', A)
print('Matrix sau khi predict:\n', print_cold_start_problem())


# In[ ]:




