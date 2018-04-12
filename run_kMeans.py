import sys
sys.path.append("F:/pythonTest")
from k_average_clustering.kMeans import *

#datMat = mat(loadDataSet('testSet.txt'))
#min(datMat[:,0])
#print(datMat[0])
#print(min(datMat[:,0]))
#print(max(datMat[:,0]))
#print(min(datMat[:,1]))
#print(max(datMat[:,1]))

#randResult = randCent(datMat, 2)
#print(randResult)

#distance = distEclud(datMat[0], datMat[1])
## print(distance)

#myCentroid, clustAssing = kMeans(datMat, 4)
#sse = sum(clustAssing[:,1])
#print(sse)
#print(myCentroids)
#print(clustAssing)

datMat3 = mat(loadDataSet('testSet2.txt'))
centList,myNewAssments = biKmeans(datMat3,3)
print("The centList is:" + str(centList))