import pandas as pd
import numpy as np
import math
from datetime import datetime
from UserCfEngine import UserCfEngine
from ItemCfEngine import ItemCfEngine
from HybridCfEngine import HybridCfEngine

ratingsPath="ratings.csv"
ratings_trainPath="ratings_training_80.csv"
ratings_tuningPath="ratings_tuning10.csv"

ratings_final_evaluationPath="ratings_validation10.csv"
ratings_final_evaluation_base_data_Path="ratings_training_temp.csv"

moviesPath="movies.csv"
ucf_analysis_Path="ucf_analysis_path.csv"
icf_analysis_Path="icf_analysis_path.csv"
hybrid_analysis_Path="hybrid_analysis_path.csv"
cluster_analysis_Path="cluster_analysis_path.csv"

maxRecords=200000
userClusters=1
itemClusters=1
ratingsThreshold=1
pccThreshold=0.1

    
def runHybridCf():
    he=HybridCfEngine(ratings_trainPath, moviesPath, maxRecords, userClusters, itemClusters, ratingsThreshold, pccThreshold)
    
    st=datetime.now()
    print("**** Hybrid CF Started at: {0} ****".format(st))
    he.initialize()
    et=datetime.now()
    print("Initialization Complete, duration: {0}, records:{1}, userClusters:{2}, itemClusters: {3}".format(et-st, maxRecords, userClusters, itemClusters))

    choice=1
    while choice!='':
        print("\nEnter userId, movieId, separated by a SPACE, or press ENTER to quit: ", end='')
        choice=input()
        if choice=='':
            break
        inList=choice.split(sep=' ')
        (userId, movieId)=(inList[0].strip(), inList[1].strip())
        pr = he.predictRatingForUserMovie(userId, movieId)
        print("Predicted Rating for user {0} movie {1} is: ucf:{2}, icf:{3}, hcf:{4}".format(userId, movieId, pr[0], pr[1], pr[2]))


def runUserCf():
    ue=UserCfEngine(ratings_trainPath, moviesPath, maxRecords, userClusters, ratingsThreshold, pccThreshold)
    st=datetime.now()
    ue.initialize()
    et=datetime.now()
    print("**** UserCF Initialization Complete, duration: {0}, records:{1}, userClusters:{2} ****".format(et-st, maxRecords, userClusters))

    choice=1
    while choice!='':
        print("\nEnter userId, movieId, separated by a SPACE, or press ENTER to quit: ", end='')
        choice=input()
        if choice=='':
            break
        inList=choice.split(sep=' ')
        (userId, movieId)=(inList[0].strip(), inList[1].strip())
        print(ue.predictRatingForUserMovie(userId, movieId))

def runItemCf():
    ie=ItemCfEngine(ratings_trainPath, moviesPath, maxRecords, itemClusters, ratingsThreshold, pccThreshold)
    st=datetime.now()
    ie.initialize()
    et=datetime.now()
    print("Initialization Complete, duration: {0}, records:{1}, itemClusters:{2}".format(et-st, maxRecords, itemClusters))


    choice=1
    while choice!='':
        print("\nEnter userId, movieId, separated by a SPACE, or press ENTER to quit: ", end='')
        choice=input()
        if choice=='':
            break
        inList=choice.split(sep=' ')
        (userId, movieId)=(inList[0].strip(), inList[1].strip())
        print(ie.predictRatingForUserMovie(userId, movieId))

def runUcfFinalEvaluation():
    st=datetime.now()
    print("**** UserCF Evaluation Started at: {0} ****".format(st))
    ue=UserCfEngine(ratings_final_evaluation_base_data_Path, moviesPath, 100000, 1, 11, pccThreshold, maxKnn=600)
    ue.initialize()
    print("**** UserCF Initialization Complete; duration: {0} ****".format(datetime.now()-st))

    eval_data=pd.read_csv(ratings_final_evaluationPath)
    analysisMap={}
    rows=eval_data.shape[0]
    print('Eval Data Count: {0}'.format(rows))
    count=0
    for i in range(rows):
        row=eval_data.iloc[i]
        userId=str(int(row["userId"]))
        movieId=str(int(row["movieId"]))
        
        obsRating=row["rating"]
        predRating=ue.predictRatingForUserMovie(userId, movieId)
        analysisMap[(userId, movieId)]=(obsRating, predRating)

    (mae,rmse)=computeRmse(analysisMap)
    record='{0},{1},{2},{3},{4},{5},{6},{7}\n'.format(ue.ratingsThreshold, ue.maxKnn, ue.pccThreshold, ue.clusters, ue.records, round(mae,4), round(rmse,4), datetime.now()-st)
    print(record, end='')
    
def runIcfFinalEvaluation():
    st=datetime.now()
    print("**** ItemCF Evaluation Started at: {0} ****".format(st))
    ie=ItemCfEngine(ratings_final_evaluation_base_data_Path, moviesPath, maxRecords=100000, clusters=1, ratingsThreshold=36, pccThreshold=pccThreshold, maxKnn=200)
    ie.initialize()
    print("**** ItemCF Initialization Complete; duration: {0} ****".format(datetime.now()-st))

    eval_data=pd.read_csv(ratings_final_evaluationPath)
    analysisMap={}
    rows=eval_data.shape[0]
    print('Eval Data Count: {0}'.format(rows))
    count=0
    for i in range(rows):
        row=eval_data.iloc[i]
        userId=str(int(row["userId"]))
        movieId=str(int(row["movieId"]))
        
        obsRating=row["rating"]
        predRating=ie.predictRatingForUserMovie(userId, movieId)
        analysisMap[(userId, movieId)]=(obsRating, predRating)

    (mae,rmse)=computeRmse(analysisMap)
    if mae is not None:
        mae=round(mae, 4)
    if rmse is not None:
        rmse=round(rmse, 4)
        
    record='{0},{1},{2},{3},{4},{5},{6},{7}\n'.format(ie.ratingsThreshold, ie.maxKnn, ie.pccThreshold, ie.clusters, ie.records, mae, rmse, datetime.now()-st)
    print(record, end='')
    
def runUcfAnalysis():
    st=datetime.now()
    fw=open(ucf_analysis_Path, mode='w')
    print("**** UserCF Analysis Started at: {0} ****".format(st))
    strHeader=('{0},{1},{2},{3},{4},{5},{6}\n'.format('ratingsThreshold', 'maxKnn', 'pccThreshold', 'userClusters', 'maxRecords', 'mae', 'rmse'))
    fw.write(strHeader)
    print(strHeader,end='')

    count=0
    for iRatingsThreshold in range(1, 51, 5):
        for iMaxKnn in range(100, 601, 100):
            ue=UserCfEngine(ratings_trainPath, moviesPath, maxRecords, userClusters, iRatingsThreshold, pccThreshold, maxKnn=iMaxKnn)
            ue.initialize()

            tuning_data=pd.read_csv(ratings_tuningPath)
            analysisMap={}
            rows=tuning_data.shape[0]
            for i in range(rows):
                row=tuning_data.iloc[i]
                userId=str(int(row["userId"]))
                movieId=str(int(row["movieId"]))
                
                obsRating=row["rating"]
                predRating=ue.predictRatingForUserMovie(userId, movieId)
                analysisMap[(userId, movieId)]=(obsRating, predRating)

            (mae,rmse)=computeRmse(analysisMap)
            record='{0},{1},{2},{3},{4},{5},{6},{7}\n'.format(iRatingsThreshold, ue.maxKnn, pccThreshold, userClusters, maxRecords, round(mae,4), round(rmse,4), datetime.now()-st)
            fw.write(record)
            print(record, end='')
            count+=1
    fw.close()
    et=datetime.now()
    print("**** UserCF Analysis Complete, duration: {0}, iterations:{1} ****".format(et-st, count))

def runIcfAnalysis():
    st=datetime.now()
    fw=open(icf_analysis_Path, mode='w')
    print("**** ItemCF Analysis Started at: {0} ****".format(st))
    strHeader=('{0},{1},{2},{3},{4},{5},{6}\n'.format('ratingsThreshold', 'maxKnn', 'pccThreshold', 'userClusters', 'maxRecords', 'mae', 'rmse'))
    fw.write(strHeader)
    print(strHeader,end='')

    count=0
    for iRatingsThreshold in range(1, 2, 1):
        for iMaxKnn in range(200, 201, 100):
            ie=ItemCfEngine(ratings_trainPath, moviesPath, maxRecords, itemClusters, iRatingsThreshold, pccThreshold, maxKnn=iMaxKnn)
            ie.initialize()

            tuning_data=pd.read_csv(ratings_tuningPath)
            analysisMap={}
            rows=tuning_data.shape[0]
            for i in range(rows):
                row=tuning_data.iloc[i]
                userId=str(int(row["userId"]))
                movieId=str(int(row["movieId"]))
                
                obsRating=row["rating"]
                predRating=ie.predictRatingForUserMovie(userId, movieId)
                analysisMap[(userId, movieId)]=(obsRating, predRating)

            (mae,rmse)=computeRmse(analysisMap)
            if mae is not None and rmse is not None:
                record='{0},{1},{2},{3},{4},{5},{6},{7}\n'.format(ie.ratingsThreshold, ie.maxKnn, ie.pccThreshold, ie.clusters, ie.records, round(mae,4), round(rmse,4), datetime.now()-st)
                fw.write(record)
                print(record, end='')
                count+=1
    fw.close()
    et=datetime.now()
    print("**** ItemCF Analysis Complete, duration: {0}, iterations:{1} ****".format(et-st, count))
    
def computeRmse(analysisMap):
    sigmaErrorSquared=0
    sigmaError=0
    n=0
    mae=None
    rmse=None
    for value in analysisMap.values():
        obsRating, predRating=(value[0], value[1])
        if predRating is not None:
            n+=1
            sigmaError+=abs(predRating-obsRating)
            sigmaErrorSquared+=(predRating-obsRating)**2
    if n>0:
        mae=sigmaError/n
        rmse=(sigmaErrorSquared/n)**0.5

    return (mae,rmse)

def runHybridAnalysis():
    st=datetime.now()
    print("**** Hybrid Analysis Started at: {0} ****".format(st))
    
    fw=open(hybrid_analysis_Path, mode='w')
    strHeader=('{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13}\n'.format('ucfCoeff', 'icfCoeff', 'maxRecords', 'uMae', 'uRmse', 'iMae', 'iRmse', 'hMae', 'hRmse', 'uExclusive', 'iExclusive', 'Both', 'Neither', 'duration'))
    fw.write(strHeader)
    print(strHeader,end='')

    tuning_data=pd.read_csv(ratings_tuningPath)
    rows=tuning_data.shape[0]
    for uCoeff in np.arange(0.69, 0.70, 0.01):
        iCoeff=1-uCoeff
        he=HybridCfEngine(ratings_trainPath, moviesPath, maxRecords=100000, ucfCoeff=uCoeff, icfCoeff=iCoeff)
        he.initialize()

        ucfMap={}
        icfMap={}
        hcfMap={}

        ucfExcl=[]
        icfExcl=[]
        neither=[]
        both=[]
        for i in range(rows):
            row=tuning_data.iloc[i]
            userId=str(int(row["userId"]))
            movieId=str(int(row["movieId"]))
            obsR=row["rating"]
            
            pr=he.predictRatingForUserMovie(userId, movieId)
            ucfMap[(userId, movieId)] = (obsR, pr[0])
            icfMap[(userId, movieId)] = (obsR, pr[1])
            hcfMap[(userId, movieId)] = (obsR, pr[2])

            
            if pr[0] is not None and pr[1] is None:
                ucfExcl.append( (userId, movieId, pr[0], pr[1], pr[2]) )
            elif pr[1] is not None and pr[0] is None:
                icfExcl.append( (userId, movieId, pr[0], pr[1], pr[2]) )
            elif pr[0] is None and pr[1] is None:
                neither.append( (userId, movieId, pr[0], pr[1], pr[2]) )
            else:
                both.append( (userId, movieId, pr[0], pr[1], pr[2]) )

        (uMae, uRmse)= computeRmse(ucfMap)
        (iMae, iRmse)= computeRmse(icfMap)
        (hMae, hRmse)= computeRmse(hcfMap)

        (uMae,uRmse)=(roundIf(uMae),roundIf(uRmse))
        (iMae,iRmse)=(roundIf(iMae),roundIf(iRmse))
        (hMae,hRmse)=(roundIf(hMae),roundIf(hRmse))
        
        record='{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13}\n'.format(uCoeff, iCoeff, he.records, uMae, uRmse, iMae, iRmse, hMae, hRmse, len(ucfExcl), len(icfExcl), len(both), len(neither), datetime.now()-st)
        fw.write(record)
        print(record, end='')
    
    fw.close()
    return
    
def roundIf(x):
    if x is not None:
        return round(x, 4)
    else:
        return None

def runClusterAnalysis():
    fw=open(cluster_analysis_Path, mode='w')
    strHeader=('{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15}\n'.format('ucfCoeff', 'icfCoeff', 'maxRecords', 'uMae', 'uRmse', 'iMae', 'iRmse', 'hMae', 'hRmse', 'uExclusive', 'iExclusive', 'Both', 'Neither', 'duration','uClusters', 'iClusters'))
    fw.write(strHeader)

    for i in range(10, 0, -1):
        record=runFinalHybridAnalysis(userClusters=i, itemClusters=1)
        fw.write(record)
        
    for i in range(10, 0, -1):
        record=runFinalHybridAnalysis(userClusters=1, itemClusters=i)
        fw.write(record)
        
    fw.close()
    return
        
def runFinalHybridAnalysis(userClusters=1, itemClusters=1):
    st=datetime.now()
    print("**** Hybrid Collaborative Filtering Recommender Evaluation Started at: {0} ****\n".format(st))
    
    he=HybridCfEngine(
        ratings_final_evaluation_base_data_Path,
        moviesPath,
        maxRecords=100000,
        ucfClusters=userClusters,
        icfClusters=itemClusters,
        ucfCoeff=0.65,
        icfCoeff=0.35)
    he.initialize()

    eval_data=pd.read_csv(ratings_final_evaluationPath)
    ucfMap={}
    icfMap={}
    hcfMap={}

    ucfExcl=[]
    icfExcl=[]
    neither=[]
    both=[]
    
    rows=eval_data.shape[0]
    count=0
    for i in range(rows):
        row=eval_data.iloc[i]
        userId=str(int(row["userId"]))
        movieId=str(int(row["movieId"]))
        
        obsR=row["rating"]
        pr=he.predictRatingForUserMovie(userId, movieId)
        ucfMap[(userId, movieId)] = (obsR, pr[0])
        icfMap[(userId, movieId)] = (obsR, pr[1])
        hcfMap[(userId, movieId)] = (obsR, pr[2])

        if pr[0] is not None and pr[1] is None:
            ucfExcl.append( (userId, movieId, pr[0], pr[1], pr[2]) )
        elif pr[1] is not None and pr[0] is None:
            icfExcl.append( (userId, movieId, pr[0], pr[1], pr[2]) )
        elif pr[0] is None and pr[1] is None:
            neither.append( (userId, movieId, pr[0], pr[1], pr[2]) )
        else:
            both.append( (userId, movieId, pr[0], pr[1], pr[2]) )

    (uMae, uRmse)= computeRmse(ucfMap)
    (iMae, iRmse)= computeRmse(icfMap)
    (hMae, hRmse)= computeRmse(hcfMap)
        
    (uMae,uRmse)=(roundIf(uMae),roundIf(uRmse))
    (iMae,iRmse)=(roundIf(iMae),roundIf(iRmse))
    (hMae,hRmse)=(roundIf(hMae),roundIf(hRmse))

    nPredicted=len(ucfExcl)+len(icfExcl)+len(both)
    print('User/Movie Rating Combinations Evaluated: {0}'.format(rows))
    print('Rating Predictions Generated: {0}'.format(nPredicted))
    print('Prediction Percentage: {0}%\n'.format(round(nPredicted/rows*100,2)))
    strHeader='System\t\tMAE\tRMSE'
    print(strHeader)
    print('-'*2*len(strHeader))
    print('User-Based CF\t{0}\t{1}'.format(uMae, uRmse))
    print('Item-Based CF\t{0}\t{1}'.format(iMae, iRmse))
    print('Hybrid CF\t{0}\t{1}'.format(hMae, hRmse))
    
    print("\n**** Hybrid Collaborative Filtering Recommender Evaluation Completed at: {0} ****\n".format(datetime.now()))
    
##    record='{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15}\n'.format(he.ucfCoeff, he.icfCoeff, he.records, uMae, uRmse, iMae, iRmse, hMae, hRmse, len(ucfExcl), len(icfExcl), len(both), len(neither), (datetime.now()-st).total_seconds(), userClusters, itemClusters)
##    print(record, end='')
    
##    return record
    return

def runOnlineRecommendations():
    he=HybridCfEngine(
        ratings_final_evaluation_base_data_Path,
        moviesPath,
        maxRecords=100000,
        ucfClusters=userClusters,
        icfClusters=itemClusters,
        ucfCoeff=0.65,
        icfCoeff=0.35)
    he.initialize()

    choice=1
    print('\n******** WELCOME TO THE GATECH FALL 2022 CS 6675 HYBRID RECOMMENDATION SYSTEM! ********')
    while choice!='':
        print("\nEnter userId to view recommendations, or press ENTER to quit: ", end='')
        choice=input()

        startTime=datetime.now()
        if choice!='':
            movieRatings = he.generateRecommendations(choice)
            movieRatings.sort(reverse=True)
            endTime=datetime.now()
            print("\nWe have {0} movie recommendations for you! Here are the TOP ones:".format(len(movieRatings)))

            print('-'*80)
            for i in range(len(movieRatings)):
                if i>25:
                    break
                movieRating = movieRatings[i]
#                print(movieRating[0], movieRating[1], he.moviesMap[movieRating[1]], sep="\t")
                print(he.moviesMap[movieRating[1]], sep="\t")
            print('-'*80)
    print("****Thank you for using the recommendation system!****")

# uncomment each line below as needed
#runUserCf()
#runUcfAnalysis()
#runUcfFinalEvaluation()
runOnlineRecommendations()
#runItemCf()
#runIcfAnalysis()
#runHybridCf()
#runHybridAnalysis()
#runFinalHybridAnalysis()
#runIcfFinalEvaluation()
#runClusterAnalysis()

