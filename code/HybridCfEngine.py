import math
import pandas as pd
from datetime import datetime
from UserCfEngine import UserCfEngine
from ItemCfEngine import ItemCfEngine

class HybridCfEngine:
    def __init__(self, ratingsPath, moviesPath, maxRecords=20000, ucfClusters=1, icfClusters=1,
                 ucfRatingsThreshold=11, icfRatingsThreshold=36, pccThreshold=0.1, ucfMaxKnn=600, icfMaxKnn=200, ucfCoeff=0.65, icfCoeff=0.35):
        self.ratingsPath=ratingsPath
        self.moviesPath=moviesPath
        self.records=maxRecords
        self.ucfClusters=ucfClusters
        self.icfClusters=icfClusters
        self.ucfRatingsThreshold=ucfRatingsThreshold
        self.icfRatingsThreshold=icfRatingsThreshold
        self.pccThreshold=pccThreshold
        self.ucfMaxKnn=ucfMaxKnn
        self.icfMaxKnn=icfMaxKnn
        self.ucfCoeff=ucfCoeff
        self.icfCoeff=icfCoeff
        self.moviesMap={}

        self.ucfEngine=UserCfEngine(ratingsPath, moviesPath, maxRecords, ucfClusters, ucfRatingsThreshold, pccThreshold, ucfMaxKnn)
        self.icfEngine=ItemCfEngine(ratingsPath, moviesPath, maxRecords, icfClusters, icfRatingsThreshold, pccThreshold, icfMaxKnn)
        
    def initialize(self):
        self.ucfEngine.initialize()
        self.icfEngine.initialize()
        self.moviesMap=self.ucfEngine.moviesMap

        return
        
    def predictRatingForUserMovie(self, userId, movieId):
        ucfRating=self.ucfEngine.predictRatingForUserMovie(userId, movieId)
        icfRating=self.icfEngine.predictRatingForUserMovie(userId, movieId)

        finalRating=None
        if (ucfRating is not None) and (icfRating is not None):
            finalRating=round(((self.ucfCoeff*ucfRating)+(self.icfCoeff*icfRating))/(self.ucfCoeff+self.icfCoeff), 2)
        elif icfRating is None:
            finalRating=ucfRating
        elif ucfRating is None:
            finalRating=icfRating

        return (ucfRating, icfRating, finalRating)

    def generateRecommendations(self, userId):
        movieRatings = []
        for movieId in self.ucfEngine.moviesMap.keys():
            if self.ucfEngine.userRatingsMap.get(userId).get(movieId)==None:
                movieRating=self.predictRatingForUserMovie(userId, movieId)[2]
                if movieRating is not None:
                    movieRatings.append((movieRating, movieId))

        return movieRatings
