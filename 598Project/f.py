# TODO: Weighted graph
# TODO: real news has more readers
# TODO: Different feature/ how many users read.. count
# TODO: 



import numpy as np
from scipy.sparse import csr_matrix
import pickle


dataset = "BuzzFeedSample182"

TRAIN_FRACTION = 0.2
TEST_FRACTION = 0.33
num_users = 0
num_news = 0
user_d = dict()
for line in open("eecs598_fd/FakeNewsNet-master/Data/BuzzFeed/BuzzFeedNewsUser.txt"):
	arr = line.rstrip().split("\t")
	news_id = int(arr[0]) - 1
	user_id = int(arr[1]) - 1
	if news_id + 1 > num_news:
		num_news = news_id + 1
	if user_id + 1 > num_users:
		num_users = user_id + 1
	if user_id not in user_d:
		user_d[user_id] = []
	# weight = int(arr[2])
	# for i in range(weight):
	user_d[user_id].append(news_id)
core_users = []
for key in user_d:
	if len(user_d[key]) > 1:
		core_users.append(key)
num_core_users = len(core_users)


# x, y, tx, ty, allx, ally, graph

num_train = int(TRAIN_FRACTION * num_news)
num_test = int(TEST_FRACTION * num_news)
# x is num_core_users dimension feature vector
# y is 2 dimension binary encoded labels
X = np.zeros((num_news, num_core_users))
Y = np.zeros((num_news, 2))
graph = dict()
for idx1 in range(num_core_users):
	user_id = core_users[idx1]
	# construct X
	temp = user_d[user_id]
	for idx2 in range(len(temp)):
		news_id = temp[idx2]
		X[news_id, idx1] = 1
	# construct bi-direction graph
	for i in temp:
		for j in temp:
			if i != j:
				if i not in graph:
					graph[i] = set() # TODO: We can try list
				graph[i].add(j)
				if j not in graph:
					graph[j] = set()
				graph[j].add(i)
for idx in range(int(num_news/2)):
	Y[idx, 0] = 1
for idx in range(int(num_news/2), num_news):
	Y[idx, 1] = 1

for key,val in graph.items():
	graph[key] = sorted(list(graph[key]))

print(X.shape)
# for i in range(len(X[44])):
# 	if X[44][i] == 1:
# 		print(i)
print(Y.shape)

# combine X, Y and shuffle
def shuffleXY(X,Y):
	idx = [[i] for i in range(len(X))]
	featureDim = X.shape[1]
	XY = np.hstack((X,Y))
	XY = np.hstack((idx,XY))
	np.random.shuffle(XY)

	X = XY[:, 1:featureDim]
	Y = XY[:, featureDim:]
	newidx = XY[:,0]
	return X,Y,newidx

# determine if train set contain both real and fake news
def containTF(Y, num_train):
	real = False
	fake = False
	for i in range(num_train):
		if Y[i][0] == 1:
			real = True
		if Y[i][1] == 1:
			fake = True
	return real and fake

# shuffle X, Y until first num_train elements in Y contain both real and fake news
X, Y, newidx = shuffleXY(X, Y)
while not containTF(Y,num_train):
	X, Y, newidx = shuffleXY(X, Y)

# modify graph according to newidx
# construct reversemap from oldidx to newidx
reverseMap = {}
for i in range(len(newidx)):
	reverseMap[newidx[i]] = i

# modify graph
modifiedGraph = {}
for key in graph:
	modifiedGraph[reverseMap[key]] = []
	for ele in graph[key]:
		modifiedGraph[reverseMap[key]].append(reverseMap[ele])

graph = modifiedGraph

# for i in range(len(X)):
# 	print(sum(X[i]))

x = X[:num_train,:]
x = csr_matrix(x)
y = Y[:num_train,:]
tx = X[-num_test:,:]
tx = csr_matrix(tx)
ty = Y[-num_test:,:]
allx = X[:-num_test,:]
allx = csr_matrix(allx)
ally = Y[:-num_test,:]

testidx = [i for i in range(num_news - num_test ,num_news)]

objects = ["x","y","tx","ty","allx","ally","graph"]

outputData = {"x":x, "y":y, "tx":tx, "ty":ty, "allx":allx, "ally":ally, "graph":graph}


for o in objects:
	with open('ind.'+dataset+"."+ o, 'wb') as f:
	    # Pickle the 'data' dictionary using the highest protocol available.
	    pickle.dump(outputData[o], f, pickle.HIGHEST_PROTOCOL)

with open('ind.'+dataset+"."+ "test.index", 'w') as f:
	for e in testidx:
		f.write(str(e)+'\n')


