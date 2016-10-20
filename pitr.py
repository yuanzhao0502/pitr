from sklearn import svm
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.cross_validation import train_test_split
##here is only for test
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.cross_validation import train_test_split
import pandas as pd
from sklearn import tree
from sklearn import cross_validation
import datetime
from sklearn.naive_bayes import GaussianNB
import numpy as np
import sys

def readfile(input):
	f=open(input)
	contents=f.readlines()
	example=[]
	protein={}
	for i in range(1,len(contents)):
		contents1=contents[i].split("\t")
		if contents1[0][:4] not in protein:
			protein[contents1[0][:4]]=contents1
		else:
			if contents1[3]!="non":
				protein[contents1[0][:4]]=contents1

	for i in protein:
		text=protein[i]
		del text[3]
		example.append(text)
	for item in example:
		for i in range(1,len(item)):
			item[i]=float(item[i])
	f.close()
	return example

def classification(example):
	f=open("training.dat","r")
	X=[]
	Y=[]
	XX=[]
	YY=[]
	r=[]
	for read in f:
		temp2=[]
		temp=read.strip().split(",")
		r.append(temp)
		tt=[]
		for i in range(1,len(temp)):
			tt.append(temp[i])
		#ttt=[1,2,4,5,6,7,8,9,10,11,12,13,18,63,64]
		#for i in ttt:
			#print temp[i]
		#	temp2.append(temp[i])
		XX.append(tt)
		YY.append(temp[-1])
	XX.remove(XX[0])
	YY.remove(YY[0])
	for i in XX:
		temp=[]
		count=0
		for j in i:
			if count<64:
				temp.append(float(j))
				count=count+1
		X.append(temp)
	for i in YY:
		Y.append(int(i))
	X=np.array(X)
	Y=np.array(Y)
	estimators = {}
	estimators['forest'] = RandomForestClassifier(n_estimators =150 ,n_jobs = -1,max_features = 'log2', min_samples_leaf = 1)
	repeat=0
	classweight=[]
	for repeat in range(100):
		print repeat
		for k in estimators.keys():
			estimators[k] = estimators[k].fit(X, Y)
			exsco=[]
			examplex=[]
			for item in example:
				examplex.append(item[1:])
			exsco=estimators[k].predict_proba(examplex)
			classweight.append(exsco)
	avg=[]
	for j in range(len(classweight[0])):
		avg.append([0,0,0,0,0,0])
	for p in range(6):
		for j in range(len(classweight[0])):

			temp1=0
			for i in range(len(classweight)):
				temp1=classweight[i][j][p]+temp1
			temp1=temp1/100
			avg[j][p]=temp1
	return avg
def writefile(output,avg,example):
	f=open(output,"w")
	f.write("Protein\tDEPTH\tGHECOM\tFPocket\tDoGSiteScorer\tIsoMif\tProACT2\n")
	for j in range(len(avg)):
		f.write(example[j][0][:4]+"\t")
		for i in range(len(avg[0])):
			f.write(str(avg[j][i]))
			f.write("\t")
		f.write("\n")

example=readfile(sys.argv[1])
avg=classification(example)
writefile(sys.argv[2],avg,example)