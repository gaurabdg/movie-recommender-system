import pandas as pd
import numpy as np
import pickle
import scipy
from scipy.sparse import csr_matrix
import os
from sklearn.model_selection import train_test_split

# Read data into Pandas DataFrame
dataset_path = '../datasets/ml-100k/u.data'
table = pd.read_table(dataset_path, sep='\t', header=None, names=['userId', 'movieId', 'rating', 'timestamp'])

# Compute various quantities
rating_num = table.shape[0]
list_movie = table['movieId'].unique()
list_user = table['userId'].unique()
users_num = len(list_user)
movies_num = len(list_movie)

print("total ratings: {}".format(rating_num))
print("total users: {}".format(len(list_user)))
print("total movies: {}".format(len(list_movie)))

# Create mappings of movie and user vs id
map_movie = {}
map_user = {}

for idx, m_id in enumerate(list_movie):
	map_movie[m_id] = idx

for idx, m_id in enumerate(list_user):
	map_user[m_id] = idx

# Split DataFrame into A(matrix to be decomposed) and test data
A_table, test_table = train_test_split(table, test_size=0.2)

# Form matrices from the respective tables
A = np.zeros([len(map_user), len(map_movie)])
test = np.zeros([len(map_user), len(map_movie)])

for i, row in A_table.iterrows():
	A[map_user[row['userId']], map_movie[row['movieId']]] = row['rating']

for i, row in test_table.iterrows():
	test[map_user[row['userId']], map_movie[row['movieId']]] = row['rating']

# print('Density:{}'.format(100.0*float(np.count_nonzero(test))/(users_num*movies_num)))
# print(A)

A = csr_matrix(A)
test = csr_matrix(test)

scipy.sparse.save_npz('../datasets/data_util/A_100k.npz', A)
scipy.sparse.save_npz('../datasets/data_util/test_100k.npz', test)

with open('../datasets/data_util/map_movie_100k.pkl', 'wb') as f:
	pickle.dump(map_movie, f)

with open('../datasets/data_util/map_user_100k.pkl', 'wb') as f:
	pickle.dump(map_user, f)