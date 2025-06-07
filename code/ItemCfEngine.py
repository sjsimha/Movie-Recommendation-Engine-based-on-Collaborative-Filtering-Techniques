import math
import pandas as pd
from datetime import datetime

class ItemCfEngine:
    def __init__(self, ratingsPath, moviesPath, maxRecords=20000, clusters=100, ratingsThreshold=1, pccThreshold=0.1, maxKnn=200):
        self.ratingsPath=ratingsPath
        self.moviesPath=moviesPath
        self.records=maxRecords
        self.clusters=clusters
        self.ratingsThreshold=ratingsThreshold
        self.pccThreshold=pccThreshold
        self.maxKnn=maxKnn
        self.movieToClusterMap={}
        self.clusterToMovieMap={}
        self.movieRatingsMap={}
        self.movieAvgRatingsMap={}
        self.movieSimMap={}

    def initialize(self):
        ratings=loadData(self.ratingsPath)
        movies = loadData(self.moviesPath)
        #moviesMap=createMoviesMap(movies)
        self.movieToClusterMap = createMovieToClusterMap(ratings.head(self.records), self.clusters)
        self.clusterToMovieMap = createClusterToMovieMap(self.movieToClusterMap)
        self.movieRatingsMap = createMovieRatingsMap(ratings.head(self.records))
        self.movieAvgRatingsMap = buildMovieAvgRatingsMap(self.movieRatingsMap)
        self.movieSimMap = buildMovieSimMap(self.clusterToMovieMap, self.movieRatingsMap, self.movieAvgRatingsMap, self.pccThreshold, self.ratingsThreshold, self.maxKnn)
        
    def predictRatingForUserMovie(self, userId, movieId):
        sigmaNumerator=sigmaDenominator=0
        knnTuples=[]
        try:
            knnTuples=self.movieSimMap[movieId]
        except KeyError:
            return None
            
        matches=0
        for knnTuple in knnTuples:
            otherMovieId=knnTuple[1]
            cos=knnTuple[0]
            try:
                otherMovieRating=self.movieRatingsMap[otherMovieId][userId]
                matches+=1
                sigmaNumerator+=cos*otherMovieRating
                sigmaDenominator+=cos
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

def createMovieToClusterMap(data, clusters):
    rows = data.shape[0]

    movieMap={}
    for i in range(rows):
        row = data.iloc[i]
        movieId = str(int(row["movieId"]))
        if not movieId in movieMap:
            movieMap[movieId]=''

    nextCluster = 1
    for key in movieMap:
        movieMap[key] = "C" + str(nextCluster)
        nextCluster = nextCluster+1
        if nextCluster > clusters:
            nextCluster = 1
            
    return movieMap

def createClusterToMovieMap(movieToClusterMap):
    clusterToMovieMap = {}
    for key, value in movieToClusterMap.items():
        if value not in clusterToMovieMap:
            clusterToMovieMap[value]=[]
        clusterToMovieMap[value].append(key)   
        
    return clusterToMovieMap

def createMovieRatingsMap(data):
    rows = data.shape[0]
    movieRatingsMap = {}
    
    for i in range(rows):
        row = data.iloc[i]
        userId = str(int(row["userId"]))
        movieId = str(int(row["movieId"]))
        rating = row["rating"]
        
        if not movieId in movieRatingsMap:
            movieRatingsMap[movieId] = {}
        movieRatingsMap[movieId][userId] = rating

    return movieRatingsMap

def buildMovieAvgRatingsMap(movieRatingsMap):
    movieAvgRatingsMap={}
    for movie, ratingsMap in movieRatingsMap.items():
        total=0.0
        avg=0.0
        n=0
        for user, rating in ratingsMap.items():
            n+=1
            total+=rating

        avg=total/n
        movieAvgRatingsMap[movie]=avg
        
    return movieAvgRatingsMap

def buildMovieSimMap(clusterToMovieMap, movieRatingsMap, movieAvgRatingsMap, pccThreshold, ratingsThreshold, maxKnn):
    movieSimMap={}
    
    for cluster, movieList in clusterToMovieMap.items():
        movieSimMap|=buildMovieSimMapForCluster(movieList, movieRatingsMap, movieAvgRatingsMap, pccThreshold, ratingsThreshold, maxKnn)
    return movieSimMap

def buildMovieSimMapForCluster(movieList, movieRatingsMap, movieAvgRatingsMap, pccThreshold, ratingsThreshold, maxKnn):
    movieSimMap={}
    returnMap={}
    for i in range(len(movieList)):
        sourceMovie=movieList[i]
        movieSimMap[sourceMovie]={}
        
        for j in range(len(movieList)):
            targetMovie=movieList[j]
            if targetMovie!=sourceMovie:
                try:
                    reverseSim=movieSimMap[targetMovie][sourceMovie]
                    movieSimMap[sourceMovie][targetMovie]=reverseSim
                except KeyError:
                    cos=computeCosine(sourceMovie, targetMovie, movieRatingsMap, movieAvgRatingsMap, ratingsThreshold)
                    
                    #use cos only if there's sufficient correlation
                    if cos>pccThreshold:
                        movieSimMap[sourceMovie][targetMovie]=cos

        knnTupleList=buildKnnTupleListForMovie(movieSimMap[sourceMovie], maxKnn)
        returnMap[sourceMovie]=knnTupleList
        
    return returnMap

def buildKnnTupleListForMovie(movieSimMap, maxKnn):
    simList=[]
    for targetMovie, cos in movieSimMap.items():
        simList.append( (cos, targetMovie) )
        
    simList.sort(reverse=True)
    returnList=[]
    for i in range(len(simList)):
        if i<maxKnn:
            returnList.append(simList[i])

    return returnList
    
def computeCosine(i, j, movieRatingsMap, movieAvgRatingsMap, ratingsThreshold):
    iMap=movieRatingsMap[i]
    jMap=movieRatingsMap[j]

    matches=0
    cos=sigmaN=sigmaDi=sigmaDj=0
        
    for userId, iRating in iMap.items():
        try:
            jRating=jMap[userId]
            
            #if we're here, it means both i and j have been rated by userId
            matches=matches+1
                
            avgI=movieAvgRatingsMap[i]
            avgJ=movieAvgRatingsMap[j]
            
            sigmaN=sigmaN+( (iRating-avgI)*(jRating-avgJ) )
            sigmaDi=sigmaDi+( (iRating-avgI)**2 )
            sigmaDj=sigmaDj+( (jRating-avgJ)**2 )
        except KeyError:
            continue

    denominator=math.sqrt(sigmaDi)*math.sqrt(sigmaDj)
    if denominator==0:
        cos=0
    else:
        cos=sigmaN/denominator

        
    if matches>=ratingsThreshold:
        discountFactor=1
    else:
        discountFactor=matches/ratingsThreshold
        
    cos=cos*discountFactor
    return cos

def buildKnnForUser(userId):
    cluster = userToClusterMap[userId]
    userPeerMap=userSimMap[cluster][userId]

    #convert dictionary to a list of tuples
    tuples=[]
    
    for key, value in userPeerMap.items():
        tuples.append((value, key))

    tuples.sort(reverse=True)
    return tuples
