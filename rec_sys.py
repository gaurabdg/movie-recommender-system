import numpy as np
from numpy import linalg as LA
import math
import time
import pandas as pd

# Reading the dataset into a Pandas DataFrame
ratings = pd.read_table('./datasets/ml-100k/u.data', sep='\t', header=None, names=['userId', 'movieId', 'rating', 'timestamp'])
ratings.drop(labels='timestamp',axis=1,inplace=True)

print("Dataset Looks like this")
print(ratings.head())

print("\nNumber of ratings present : " + str(len(ratings.values)))  


# Converting dataframe to equivalent Utility Matrix


def makeUtilityMatrix(ratings, verbose=True):
    """
    Generates the utility matrix from the data frame
    :param ratings
    :param verbose
    :return: ratingsMatrix, movieAlises
    """
    intermediateMoviesMatrix = np.array(ratings.values)
    usersList = intermediateMoviesMatrix[:,0].astype(int)
    moviesList = intermediateMoviesMatrix[:,1].astype(int)
    if verbose:
        print("Min User ID : "   + str(usersList.min()) +               "\nMax User ID : " + str(usersList.max()) +               "\nUnique User IDs : " + str(len(np.unique(usersList)))              )

        print("")

        print("Min Movie ID : "   + str(moviesList.min()) +               "\nMax Movie ID : " + str(moviesList.max()) +               "\nUnique Movie IDs : " + str(len(np.unique(moviesList)))              )

    movieAliasNext = 0
    movieAlises = dict()
    for i in moviesList:
        if i not in movieAlises:
            movieAlises[i] = movieAliasNext
            movieAliasNext += 1
    
    ## Creating a utility matrix
    ratingsMatrix = np.zeros(shape=(len(np.unique(usersList)), len(np.unique(moviesList))))
    for row in ratings.values:
        ## print(row)
        userID = int(row[0])
        movieID = int(row[1])
        rating  = row[2]
        ratingsMatrix[userID-1][movieAlises[movieID]] = rating
    print(ratingsMatrix)
    return ratingsMatrix,movieAlises

ratingsMatrix,_ = makeUtilityMatrix(ratings)


def validationData(df, split = 0.8):
    """
    Split dataset
    :param df: Pandas dataframe
    :param split: The ratio in which to split train and test data
    :return: ratingsMatrix, testUserMoviePairList
    """
    ratingsMatrix,alias = makeUtilityMatrix(df,False)
    
    mask = np.random.rand(len(df)) < split
    test = df[~mask]
    testUserMoviePairList = []
    for t in test.values:
        testUserMoviePairList.append( (int(t[0]-1),alias[int(t[1])]))
        #ratingsMatrix[t[0]][t[1]] = 0
    return (ratingsMatrix, testUserMoviePairList)

testRatingMatrix, testPairList = validationData(ratings,split=0.60)
print("Number of items for validation : " + str(len(testPairList)))


# Generating Similarity Matrix
# Both user user and item item is supported by this function. Optimised to run fast

def generateSimilarityMatrix(utilityMatrix,similarityBetween="user-user", center=True):
    """
    We are generating user user similarity using the Pearson Correlation Coefficient. if center = True
    This ensures that the generous raters and strict raters are handled appropriately.
    :param utilityMatrix
    :param similarityBetween: specify the users
    :param center: centred cosine
    :return: similarity matrix
    """
    assert type(utilityMatrix) == np.ndarray
    assert similarityBetween == "user-user" or similarityBetween == 'item-item'
    
    utilityMatrix = np.copy(utilityMatrix)
    
    if similarityBetween == 'item-item':
        utilityMatrix = np.transpose(utilityMatrix)
    
    numUsers = utilityMatrix.shape[0]
    numItems = utilityMatrix.shape[1]
    if center:
        for i in range(numUsers):
            nonZero = 0
            rollingSum = 0
            for j in range(numItems):
                if utilityMatrix[i][j] != 0:
                    nonZero += 1
                    rollingSum += utilityMatrix[i][j]

            if(nonZero > 0):
                centreVal = rollingSum / nonZero
            else:
                centreVal = 0

            for j in range(numItems):
                if utilityMatrix[i][j] != 0:
                    utilityMatrix[i][j] -= centreVal

    unitizedUtilityMatrix = np.zeros(shape = (numUsers,numItems), dtype=np.longdouble)
    for i in range(numUsers):
        length = 0.0
        for j in range(numItems):
            length += utilityMatrix[i][j] * utilityMatrix[i][j]
        
        length = math.sqrt(length)
        for j in range(numItems):
            if(length != 0):
                unitizedUtilityMatrix[i][j] = utilityMatrix[i][j] / length
            else:
                unitizedUtilityMatrix[i][j] = 0
                #print("Possible loss of accuracy. Numerator is :  " + str(unitizedUtilityMatrix[i][j]))

    ''' 
        Numpy Matrix multiplication was much faster than the normal usage. This has been 
        experimentally verified
    '''
    similarityMatrix = np.dot(unitizedUtilityMatrix, np.transpose(unitizedUtilityMatrix))
    return similarityMatrix


# Returns indices, sorted from the ones with the highest similarity to the lowest. Doesnt remove itself

def sortedBySimilarity(similarityArray):
    """
    This function looks at the the similarity scores and return the indexes in a descending order by similarity
    :param similarityArray:
    :return: indices
    """
    assert type(similarityArray) == np.ndarray and len(similarityArray.shape) == 1 
    similarityArray = np.copy(similarityArray)
    almostRet = []
    for i in range(len(similarityArray)):
        almostRet.append((i,similarityArray[i]))
    
    almostRet = sorted(almostRet,key=lambda x : x[1],reverse=True)
    ret = []
    for i in almostRet:
        ret.append(i[0])
    
    return ret


# Finding Global Baseline Parameters

def getBaselineParameters(utilityMatrix):
    """
    Compute baseline parameters
    :param utilityMatrix
    :return: globalAverage, userDeviations, movieDeviations
    """
    userAverages  = []
    movieAverages = []
    globalSumRatings = 0
    globalNumRatedMovies = 0
    for userRatings in utilityMatrix:
        numRatedMovies = 0
        sumRatings = 0
        
        for j in userRatings:
            if j != 0:
                numRatedMovies += 1
                sumRatings += j
                globalNumRatedMovies += 1
                globalSumRatings += j
     
        userAverages.append(sumRatings/numRatedMovies)
        
    for movieRatings in utilityMatrix.T:
        numRatedMovies = 0
        sumRatings = 0
        
        for j in movieRatings:
            if j != 0:
                numRatedMovies += 1
                sumRatings += j
        movieAverages.append(sumRatings / numRatedMovies)
    
    globalAverage = globalSumRatings / globalNumRatedMovies
    
    userDeviations = userAverages - globalAverage
    movieDeviations = movieAverages - globalAverage

    return (globalAverage, userDeviations, movieDeviations)


# Recommender System using Collaborative Filtering

# User User Collaborative Filtering

def userUserCF(utilityMatrix, similarityMatrix,userItemPredictList, k = 1):
    """
    Computes user user collaborative filtering
    :param utilityMatrix
    :param similarityMatrix
    :param userItemPredictList
    :param k
    :return: user item rating list
    """
    userToPredictWith = dict()
    for pair in userItemPredictList:
        user = pair[0]
        item = pair[1]
        if user not in userToPredictWith:
            userToPredictWith[user] = [item]
        else:
            userToPredictWith[user].append(item)
    
    returnList = []
    
    for user in userToPredictWith:
        similarityIndex = sortedBySimilarity(similarityMatrix[user])
        similarityIndex.remove(user)
        for item in userToPredictWith[user]:
            similarityRatingList = []
            rating = 0.0
            sumSimilarities = 0.0
            for similarUser in similarityIndex:
                if utilityMatrix[similarUser][item] != 0: 
                        similarityRatingList.append( (similarityMatrix[user][similarUser],                                                     utilityMatrix[similarUser][item]) )
                if len(similarityRatingList) >= k:
                    break
            for p in similarityRatingList:
                rating += p[0]*p[1]
                sumSimilarities += p[0]
            if sumSimilarities != 0:
                rating = rating / sumSimilarities
            else:
                rating = 0
            returnList.append( (user,item,rating) )
    return returnList


# Item Item Collaborative Filtering

def itemItemCF(utilityMatrix, similarityMatrix, userItemPredictList, k=1):
    """
    Computes item item collaborative filtering
    :param utilityMatrix
    :param similarityMatrix
    :param userItemPredictList
    :param k:
    :return: rating list
    """
    itemsToPredictWith = dict()
    for pair in userItemPredictList:
        user = pair[0]
        item = pair[1]
        if item not in itemsToPredictWith:
            itemsToPredictWith[item] = [user]
        else:
            itemsToPredictWith[item].append(user)
    
    returnList = []
    
    for item in itemsToPredictWith:
        similarityIndex = sortedBySimilarity(similarityMatrix[item])
        
        for user in itemsToPredictWith[item]:
            similarityRatingList = []
            rating = 0.0
            sumSimilarities = 0.0
            for similarItem in similarityIndex:
                if utilityMatrix[user][similarItem] != 0:
                    similarityRatingList.append( (similarityMatrix[item][similarItem],                                                 utilityMatrix[user][similarItem]) )
                if len(similarityRatingList) >= k:
                    break
            for p in similarityRatingList:
                rating += p[0]*p[1]
                sumSimilarities += p[0]
            if sumSimilarities != 0:
                rating = rating / sumSimilarities
            else:
                rating = 0
            returnList.append( (user,item,rating) )
    return returnList


# Collaborative Filtering with Global Baseline

def baselineCF(utilityMatrix, similarityMatrix, userItemPredictList, k=1):
    """
    Computes collaborative filtering with baseline approach
    :param utilityMatrix
    :param similarityMatrix
    :param userItemPredictList
    :param k
    """
    globalAverage, userDeviations, movieDeviations = getBaselineParameters(utilityMatrix)
    matrixShape = utilityMatrix.shape
    base = np.zeros(shape=matrixShape)
    baselineUtilityMatrix = base + movieDeviations
    baselineUtilityMatrix = baselineUtilityMatrix.T
    baselineUtilityMatrix = baselineUtilityMatrix + userDeviations
    baselineUtilityMatrix = baselineUtilityMatrix.T
    
    alternateUtility = utilityMatrix - baselineUtilityMatrix
    
    ret = itemItemCF(alternateUtility, similarityMatrix, userItemPredictList,k)
    retNew = []
    for r in ret:
        retNew.append( (r[0], r[1], r[2] + baselineUtilityMatrix[r[0]][r[1]]) )
    
    return retNew


# ## Singular Value Decomposition

def energy(sigma, reqd_energy):
    """
    Calculates the number of singular values to retain in order to ensure the specified energy is retained.
    :param sigma: singular values array
    :param reqd_energy: energy to retain
    :return: num_singular_vals
    """
    if sigma.ndim == 2:
        sigma = np.squeeze(sigma)
    if reqd_energy == 0:
        return -1
    # calculate total sum
    total_energy = np.sum(sigma)

    # calculate percent energy
    percentage_reqd_energy = reqd_energy * total_energy / 100.0

    # calculate cumulative sum of the singular values in non-decreasing order
    indexSum = sigma.cumsum()
    checkSum = indexSum <= percentage_reqd_energy

    # number of singular values to consider
    last_singular_val_pos = np.argmin(checkSum)
    num_singular_vals = last_singular_val_pos + 1
    return num_singular_vals

def compute_svd(A, energy_retain=90):
    """
    :param A: matrix to be decomposed
    :param energy_retain: energy to retain
    :return: U, sigma, Vtr
    """
    # get eigen values and vectors
    eig_vals, eig_vecs = LA.eig( np.dot(A.T, A))
    eig_vals = np.absolute(np.real(eig_vals))
    #print('eigen values: {}\n\neigen vectors: {}'.format(eig_vals, eig_vecs))
    
    # calculate the number of eigen values to retain
    if energy_retain == 100:
        eig_vals_num = LA.matrix_rank(np.dot(A.T, A))
    else:
        # sort eigen values in increasing order and compute the number of eigen values to be retained
        eig_vals_num = energy(np.sort(eig_vals)[::-1], energy_retain)

    ##print('No of eigenvalues retained:{}'.format(eig_vals_num))
    # place the eigen vectors according to increasing order of their corresponding eigen values to form V
    eig_vecs_num = np.argsort(eig_vals)[::-1][0:eig_vals_num]  # TODO
    V = np.real(eig_vecs[:, eig_vecs_num])
    
    # Calculation of sigma | sort in decreasing order and fill till number of eigen values to retain
    sigma_vals = np.reshape(np.sqrt(np.sort(eig_vals)[::-1])[0:eig_vals_num], eig_vals_num)
    sigma = np.zeros([eig_vals_num, eig_vals_num])
    np.fill_diagonal(sigma, sigma_vals)

    # Calculation of U by using U = AVS^-1
    U = np.dot(A, np.dot(V, LA.inv(sigma)))

    Vtr = V.T
    #print("U: {}".format(U))
    #print("sigma: {}".format(sigma))
    #print("V_transpose: {}".format(Vtr))
    return U, sigma, Vtr


# ## CUR decomposition

def row_col_selection(A, r, repeat):
    """
    Function for row/column selection from A to form R/C respectively
    :param A: matrix to be decomposed
    :param r: number of selections
    :param repeat: is repetitive selection allowed or not
    :return: selected rows, R
    """
    index_set = [i for i in range(len(A))]
    frob = 0

    # compute frobenius norm for A
    for i in range(len(A)):
        for j in range(len(A[i])):
            frob += A[i][j] ** 2

    # compute prob for random selection of rows and columns
    prob = np.zeros(len(A))
    for i in range(len(A)):
        sum_sqr_row_vals = 0
        for j in range(len(A[i])):
            sum_sqr_row_vals += A[i][j]**2
        prob[i] = sum_sqr_row_vals / float(frob)

    sel_rows = np.random.choice(index_set, r, repeat, prob)

    # form R/C with random selected rows/columns
    R = np.zeros((r, len(A[0])))
    for i, row in zip(range(r), sel_rows):
        for j in range(len(A[row])):
            R[i][j] = A[row][j]
            R[i][j] = R[i][j]/float(math.sqrt(r*prob[row]))

    return sel_rows, R


def compute_U(A, r, row_idx, col_idx, ret_energy):
    """
    Computation of U using Moore-Pennrose pseudoinverse
    :param A: matrix to be decomposed
    :param r: number of row/col selection
    :param row_idx: set of selected row indices
    :param col_idx: set of selected column indices
    :return: U
    """
    # Form W by intersection of C and R
    W = np.zeros((r, r))
    for i, row in zip(range(len(row_idx)), row_idx):
        for j, column in zip(range(len(col_idx)), col_idx):
            W[i][j] = A[row][column]

    # Compute pseudo-inverse of W
    X, sigma, Ytr = compute_svd(W, ret_energy)
    sig_plus = np.zeros((sigma.shape[0], sigma.shape[1]))

    # replace non-zero sigma values with its inverse
    for i in range(len(sigma)):
        if sigma[i][i] != 0:
            sig_plus[i][i] = 1/float(sigma[i][i])

    # finally compute U with the given formula
    U = np.dot(np.dot(Ytr.T, np.dot(sig_plus, sig_plus)), X.T)

    return U


def compute_cur(A, r, ret_energy):
    """
    Main function to comput C, U, R
    :param A: matrix to be decomposed
    :param r: number of row/col selection
    :return: C, U, R
    """
    row_idx, tmpR = row_col_selection(A, r, False)
    col_idx, tmpC = row_col_selection(A.T, r, False)
    R = tmpR
    C = tmpC.T

    # print(R)
    U = compute_U(A, r, row_idx, col_idx, ret_energy)
    #print(np.dot(np.dot(C, U), R))

    return C, U, R


# ## Evaluation of Methods

def rsme(utilityMatrix, predictedValuesTupleList):
    """
    calculates Root Mean Square Error
    :param utilityMatrix
    :param predictedValuesTupleList
    :return: result of the formula
    """
    n = len(predictedValuesTupleList)
    sumSquaredDistances = 0.00
    for tup in predictedValuesTupleList:
        user = tup[0]
        item = tup[1]
        rating = tup[2]
        diff = utilityMatrix[user][item] - rating
        sumSquaredDistances += diff*diff
    sumSquaredDistances = sumSquaredDistances / n
    return math.sqrt(sumSquaredDistances)


def spearman(utilityMatrix, predictedValuesTupleList):
    """
    Calculates the Spearman Rank Correlation Coefficient
    :param utilityMatrix:
    :param predictedValuesTupleList:
    :return: result
    """
    n = len(predictedValuesTupleList)
    sumSquaredDistances = 0.00
    for tup in predictedValuesTupleList:
        user = tup[0]
        item = tup[1]
        rating = tup[2]
        diff = utilityMatrix[user][item] - rating
        sumSquaredDistances += diff*diff
    spearmanCoeff = 1.0 - 6.0*sumSquaredDistances/(n*(n-1)*(n+1))
    return spearmanCoeff


def precisionAtK(utilityMatrix, predictedValuesTupleList, k = 35000, recommedCutOff = 2.5):
    """
    Calculates precision at k
    :param utilityMatrix
    :param predictedValuesTupleList
    :param k
    :param recommedCutOff
    :return: result
    """
    #almostRet = sorted(almostRet,key=lambda x : x[1],reverse=True)
    predictedValuesTupleList = sorted(predictedValuesTupleList, key=lambda x : x[2], reverse = True)
    systemRecommends = 0
    realRecommends = 0
    for i in range(k):
        userId = predictedValuesTupleList[i][0]
        itemId = predictedValuesTupleList[i][1]
        testRec = predictedValuesTupleList[i][2] > recommedCutOff
        realRec = utilityMatrix[userId][itemId] > recommedCutOff
        if testRec:
            systemRecommends += 1
    return systemRecommends / k


# Running all recommender systems with statistics

### testRatingMatrix is the test utility matrix
### ratingsMatrix is the actual utility matrix
k = 3


# (User User) Collaborative Filtering (without handling strict and generous raters)

print("(User User) Collaborative Filtering (without handling strict and generous raters)")
start = time.process_time()
uusim = generateSimilarityMatrix(testRatingMatrix, 'user-user', center= False)
predicts = userUserCF(testRatingMatrix,uusim,testPairList,k)
end = time.process_time()
print("CPU Time : " + str(end-start))
rsmeVal = rsme(ratingsMatrix, predicts)
print("RSME     = " + str(rsmeVal))
spearmanVal = spearman(ratingsMatrix,predicts)
print("Spearman = " + str(spearmanVal))
patk = precisionAtK(ratingsMatrix, predicts)
print("Prec@K   : " + str(patk))


#  (User User) Collaborative Filtering (handling strict and generous raters)

print("(User User) Collaborative Filtering (handling strict and generous raters)")
start = time.process_time()
uusim = generateSimilarityMatrix(testRatingMatrix, 'user-user')
predicts = userUserCF(testRatingMatrix,uusim,testPairList,k)
end = time.process_time()
print("CPU Time : " + str(end-start))
rsmeVal = rsme(ratingsMatrix, predicts)
print("RSME     = " + str(rsmeVal))
spearmanVal = spearman(ratingsMatrix,predicts)
print("Spearman = " + str(spearmanVal))
patk = precisionAtK(ratingsMatrix, predicts)
print("Prec@K   : " + str(patk))


# (Item Item) Collaborative Filtering (without handling generous and strict raters)


print("(Item Item) Collaborative Filtering (without handling generous and strict raters)")
start = time.process_time()
iisim = generateSimilarityMatrix(testRatingMatrix, 'item-item', center= False)
predicts = itemItemCF(testRatingMatrix,iisim,testPairList,k)
end = time.process_time()
print("CPU Time : " + str(end-start))
rsmeVal = rsme(ratingsMatrix, predicts)
print("RSME     = " + str(rsmeVal))
spearmanVal = spearman(ratingsMatrix,predicts)
print("Spearman = " + str(spearmanVal))
patk = precisionAtK(ratingsMatrix, predicts)
print("Prec@K   : " + str(patk))


# (Item Item) Collaborative Filtering (handling generous and strict raters)

print("(Item Item) Collaborative Filtering (handling generous and strict raters)")
start = time.process_time()
iisim = generateSimilarityMatrix(testRatingMatrix, 'item-item')
predicts = itemItemCF(testRatingMatrix,iisim,testPairList,k)
end = time.process_time()
print("CPU Time : " + str(end-start))
rsmeVal = rsme(ratingsMatrix, predicts)
print("RSME     = " + str(rsmeVal))
spearmanVal = spearman(ratingsMatrix,predicts)
print("Spearman = " + str(spearmanVal))
patk = precisionAtK(ratingsMatrix, predicts)
print("Prec@K   : " + str(patk))


# (Item Item) Collaborative Filtering with Baseline

print("(Item Item) Collaborative Filtering with Baseline")
start = time.process_time()
iisim = generateSimilarityMatrix(testRatingMatrix, 'item-item')
predicts = baselineCF(testRatingMatrix,iisim,testPairList,k)
end = time.process_time()
print("CPU Time : " + str(end-start))
rsmeVal = rsme(ratingsMatrix, predicts)
print("RSME     = " + str(rsmeVal))
spearmanVal = spearman(ratingsMatrix,predicts)
print("Spearman = " + str(spearmanVal))
patk = precisionAtK(ratingsMatrix, predicts)
print("Prec@K   : " + str(patk))


# Recommendation using SVD to generate Item Item similarity (with 100% energy)

print("Recommendation using SVD to generate Item Item similarity (with 100% energy)")
start = time.process_time()
_,__,vtr = compute_svd(testRatingMatrix, energy_retain =100)
inbetween = time.process_time()
print("CPU Time for SVD: " + str(inbetween - start))
iisim = generateSimilarityMatrix(vtr,'item-item',center=False)
predicts = itemItemCF(testRatingMatrix,iisim,testPairList,k)
end = time.process_time()
print("CPU Time : " + str(end-start))
rsmeVal = rsme(ratingsMatrix, predicts)
print("RSME     = " + str(rsmeVal))
spearmanVal = spearman(ratingsMatrix,predicts)
print("Spearman = " + str(spearmanVal))
patk = precisionAtK(ratingsMatrix, predicts)
print("Prec@K   : " + str(patk))


# Recommendation using SVD to generate Item Item similarity (with 90% energy)

print("Recommendation using SVD to generate Item Item similarity (with 90% energy)")
start = time.process_time()
_,__,vtr = compute_svd(testRatingMatrix, energy_retain =90)
inbetween = time.process_time()
print("CPU Time for SVD: " + str(inbetween - start))
iisim = generateSimilarityMatrix(vtr,'item-item',center=True)
predicts = itemItemCF(testRatingMatrix,iisim,testPairList,k)
end = time.process_time()
print("CPU Time : " + str(end-start))
rsmeVal = rsme(ratingsMatrix, predicts)
print("RSME     = " + str(rsmeVal))
spearmanVal = spearman(ratingsMatrix,predicts)
print("Spearman = " + str(spearmanVal))
patk = precisionAtK(ratingsMatrix, predicts)
print("Prec@K   : " + str(patk))


# Recommendation using CUR to generate Item Item similarity (with 100% energy)

print("Recommendation using CUR to generate Item Item similarity (with 100% energy)")
start = time.process_time()
C, U, R = compute_cur(testRatingMatrix, r=470, ret_energy=100)
iisim = generateSimilarityMatrix(R,'item-item',center=True)
predicts = itemItemCF(testRatingMatrix,iisim,testPairList,k)
end = time.process_time()
print("CPU Time : " + str(end-start))
rsmeVal = rsme(ratingsMatrix, predicts)
print("RSME     = " + str(rsmeVal))
spearmanVal = spearman(ratingsMatrix,predicts)
print("Spearman = " + str(spearmanVal))
patk = precisionAtK(ratingsMatrix, predicts)
print("Prec@K   : " + str(patk))


# Recommendation using CUR to generate Item Item similarity (with 90% energy)

print("Recommendation using CUR to generate Item Item similarity (with 90% energy)")
start = time.process_time()
C, U, R = compute_cur(testRatingMatrix, r=470, ret_energy=90)
iisim = generateSimilarityMatrix(R,'item-item',center=True)
predicts = itemItemCF(testRatingMatrix,iisim,testPairList,k)
end = time.process_time()
print("CPU Time : " + str(end-start))
rsmeVal = rsme(ratingsMatrix, predicts)
print("RSME     = " + str(rsmeVal))
spearmanVal = spearman(ratingsMatrix,predicts)
print("Spearman = " + str(spearmanVal))
patk = precisionAtK(ratingsMatrix, predicts)
print("Prec@K   : " + str(patk))

