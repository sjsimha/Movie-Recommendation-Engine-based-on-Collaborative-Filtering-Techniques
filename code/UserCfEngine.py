import math
import pandas as pd
from datetime import datetime
#ratingsPath = loadData("C:/Users/Sharath/OneDrive/Documents/CS 6675/Project/ml-latest-small/ml-latest-small/ratings.csv")
#moviesPath = loadData("C:/Users/Sharath/OneDrive/Documents/CS 6675/Project/ml-latest-small/ml-latest-small/movies.csv")

class UserCfEngine:
    def __init__(self, ratingsPath, moviesPath, maxRecords=20000, clusters=10, ratingsThreshold=1, pccThreshold=0.1, maxKnn=200):
        self.ratingsPath=ratingsPath
        self.moviesPath=moviesPath
        self.records=maxRecords
        self.clusters=clusters
        self.ratingsThreshold=ratingsThreshold
        self.pccThreshold=pccThreshold
        self.maxKnn=maxKnn
        self.userToClusterMap={}
        self.clusterToUserMap={}
        self.userRatingsMap={}
        self.userAvgRatingsMap={}
        self.userSimMap={}
        self.moviesMap={}

    def initialize(self):
        ratings=loadData(self.ratingsPath)
        movies = loadData(self.moviesPath)
        self.moviesMap=createMoviesMap(movies)
        self.userToClusterMap = createUserToClusterMap(ratings.head(self.records), self.clusters)
        self.clusterToUserMap = createClusterToUserMap(self.userToClusterMap)
        self.userRatingsMap = createUserRatingsMap(ratings.head(self.records))
        self.userAvgRatingsMap = buildUserAvgRatingsMap(self.userRatingsMap)
        self.userSimMap = buildUserSimMap(self.clusterToUserMap, self.userRatingsMap, self.userAvgRatingsMap, self.pccThreshold, self.ratingsThreshold, self.maxKnn)

        return
        
    def predictRatingForUserMovie(self, userId, movieId):
        sigmaNumerator=sigmaDenominator=0
        cluster=''
        knnTuples=[]

        try:
            cluster=self.userToClusterMap[userId]
            knnTuples=self.userSimMap[cluster][userId]
        except KeyError:
            return None

        matches=0
        for knnTuple in knnTuples:
            otherUserId=knnTuple[1]
            pcc=knnTuple[0]
            try:
                otherUserRating=self.userRatingsMap[otherUserId][movieId]
                matches+=1
                sigmaNumerator+=pcc*(otherUserRating-self.userAvgRatingsMap[otherUserId])
                sigmaDenominator+=pcc
            except KeyError:
                continue
        if sigmaNumerator==0 or sigmaDenominator==0:
            return None
        else:
            predictedRating = round(self.userAvgRatingsMap[userId] + sigmaNumerator/sigmaDenominator, 2)
            return predictedRating

    def predictMovieRatingForUserCf(userRatingsMap, userId, movieId, knnTuples):
        sigmaNumerator=sigmaDenominator=0
        for knnTuple in knnTuples:
            otherUserId=knnTuple[1]
            pcc=knnTuple[0]
            try:
                otherUserRating=userRatingsMap[otherUserId][movieId]
                sigmaNumerator+=pcc*otherUserRating
                sigmaDenominator+=pcc
            except KeyError:
                continue
        if sigmaNumerator==0 or sigmaDenominator==0:
            return None
        else:
            predictedRating = round(sigmaNumerator/sigmaDenominator, 2)
            return predictedRating

    def generateRecommendations(userId):
        tuples = buildKnnForUser(userId)
        userMovieRatingsMap = userRatingsMap.get(userId)
        
        movieRatings = []
        for movieId in moviesMap.keys():
            if userMovieRatingsMap.get(movieId)==None:
                movieRating=predictMovieRatingForUserCf(userRatingsMap, userId, movieId, tuples)
                if movieRating is not None:
                    movieRatings.append((movieRating, movieId))

        return movieRatings
    
def runOnlineRecommendations():
    choice=1
    while choice!='':
        print("\nEnter userId to view recommendations, or press ENTER to quit: ", end='')
        choice=input()

        startTime=datetime.now()
        if choice!='':
            movieRatings = generateRecommendations(choice)
            movieRatings.sort(reverse=True)
            endTime=datetime.now()
            print("We have {0} movie recommendations for you. Duration is {1}. Here are the top ones\n".format(len(movieRatings), endTime-startTime))

            for i in range(len(movieRatings)):
                if i>25:
                    break
                movieRating = movieRatings[i]
                print(movieRating[0], movieRating[1], moviesMap[movieRating[1]], sep="\t")
    print("\n****Thank you for using the recommendation system!****")

def loadData(filePath):
    data = pd.read_csv(filePath)
    return data

def createMoviesMap(data):
    rows=data.shape[0]
    moviesMap={}
    for i in range(rows):
        row = data.iloc[i]
        movieId = str(int(row["movieId"]))
        title = row["title"]
        moviesMap[movieId]=title

    return moviesMap

def createUserToClusterMap(data, clusters):
    rows = data.shape[0]
    userMap={}
    for i in range(rows):
        row = data.iloc[i]
        userId = str(int(row["userId"]))
        if not userId in userMap:
            userMap[userId]=''

    nextCluster = 1
    for key in userMap:
        userMap[key] = "C" + str(nextCluster)
        nextCluster = nextCluster+1
        if nextCluster > clusters:
            nextCluster = 1
            
    return userMap

def createClusterToUserMap(userToClusterMap):
    clusterToUserMap = {}
    for key, value in userToClusterMap.items():
        if value not in clusterToUserMap:
            clusterToUserMap[value]=[]
        clusterToUserMap[value].append(key)   
        
    return clusterToUserMap

def createUserRatingsMap(data):
    rows = data.shape[0]
    userRatingsMap = {}
    
    for i in range(rows):
        row = data.iloc[i]
        userId = str(int(row["userId"]))
        movieId = str(int(row["movieId"]))
        rating = row["rating"]
        
        if not userId in userRatingsMap:
            userRatingsMap[userId] = {}
        userRatingsMap[userId][movieId] = rating

    return userRatingsMap

def buildUserAvgRatingsMap(userRatingsMap):
    userAvgRatingsMap={}
    for user, ratingsMap in userRatingsMap.items():
        total=0.0
        avg=0.0
        n=0
        for item, rating in ratingsMap.items():
            n=n+1
            total=total+rating

        avg=total/n
        userAvgRatingsMap[user]=avg
        
    return userAvgRatingsMap

def buildUserSimMap(clusterToUserMap, userRatingsMap, avgUserRatingsMap, pccThreshold, ratingsThreshold, maxKnn):
    userSimMap={}
    for cluster, userList in clusterToUserMap.items():
        userSimMap[cluster] = buildUserSimMapForCluster(userList, userRatingsMap, avgUserRatingsMap, pccThreshold, ratingsThreshold, maxKnn)
        
    return userSimMap

def buildUserSimMapForCluster(userList, userRatingsMap, avgUserRatingsMap, pccThreshold, ratingsThreshold, maxKnn):
    userSimMap={}
    returnMap={}
    for i in range(len(userList)):
        sourceUser=userList[i]
        userSimMap[sourceUser]={}
        
        for j in range(len(userList)):
            targetUser=userList[j]
            if targetUser!=sourceUser:
                try:
                    reverseSim=userSimMap[targetUser][sourceUser]
                    userSimMap[sourceUser][targetUser]=reverseSim
                except KeyError:
                    pcc=computePcc(sourceUser, targetUser, userRatingsMap, avgUserRatingsMap, ratingsThreshold)
                    
                    #use pcc only if there's sufficient correlation
                    if pcc>0.1:
                        userSimMap[sourceUser][targetUser]=pcc

        knnTupleList=buildKnnTupleListForUser(userSimMap[sourceUser], maxKnn)
        returnMap[sourceUser]=knnTupleList
        
    return returnMap

def buildKnnTupleListForUser(userSimMap, maxKnn):
    simList=[]
    for targetUser, pcc in userSimMap.items():
        simList.append( (pcc, targetUser) )
        
    simList.sort(reverse=True)
    returnList=[]
    for i in range(len(simList)):
        if i<maxKnn:
            returnList.append(simList[i])

    return returnList
    
def computePcc(u, v, userRatingsMap, avgRatingsMap, ratingsThreshold):
    uMap=userRatingsMap[u]
    vMap=userRatingsMap[v]

    matches=0
    sigmaN=sigmaDu=sigmaDv=0
        
    for movieId, uRating in uMap.items():
        try:
            vRating=vMap[movieId]
            
            #if we're here, it means both u and v have rated movieId
            matches=matches+1
            avgU=avgRatingsMap[u]
            avgV=avgRatingsMap[v]
            sigmaN=sigmaN+( (uRating-avgU)*(vRating-avgV) )
            sigmaDu=sigmaDu+( (uRating-avgU)**2 )
            sigmaDv=sigmaDv+( (vRating-avgV)**2 )
        except KeyError:
            continue

    denominator=math.sqrt(sigmaDu)*math.sqrt(sigmaDv)
    if denominator==0:
        pcc=0
    else:
        pcc=sigmaN/denominator

    if matches>=ratingsThreshold:
        discountFactor=1
    else:
        discountFactor=matches/ratingsThreshold
        
    pcc=pcc*discountFactor
    return pcc

def buildKnnForUser(userId):
    cluster = userToClusterMap[userId]
    userPeerMap=userSimMap[cluster][userId]

    #convert dictionary to a list of tuples
    tuples=[]
    
    for key, value in userPeerMap.items():
        tuples.append((value, key))

    tuples.sort(reverse=True)
    return tuples

