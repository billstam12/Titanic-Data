import string
import numpy as np
import pandas as pd

################################## PREPROCESSING ########################################
train_set = pd.read_csv("datasets/train.csv")

def substrings_in_string(big_string, substrings):
	for substring in substrings:
		if string.find(big_string, substring) != -1:
			return substring

def replace_titles(x):
	title = x['Title']
	if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
		return 'Mr'
	elif title in ['Countess', 'Mme']:
		return 'Mrs'
	elif title in ['Mlle', 'Ms']:
		return 'Miss'
	elif title =='Dr':
		if x['Sex']=='Male':
			return 'Mr'
		else:
			return 'Mrs'
	else:
		return title

# Turn name into Title
		
title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                    'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
                    'Don', 'Jonkheer']

train_set['Title'] = train_set["Name"].map(lambda x: substrings_in_string(x, title_list))
train_set['Title'] = train_set.apply(replace_titles, axis =  1)

# Turn cabin into deck

train_set.Cabin = train_set.Cabin.fillna('Unknown')    
cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']
train_set['Deck']=train_set['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))
	

# Family 

train_set['Family_Size'] = train_set["SibSp"] + train_set["Parch"]

# Characteristics

train_set['Age*Class']=train_set['Age']*train_set['Pclass']

# Fare per person

train_set["Fare_per_person"] = train_set["Fare"]/(train_set["Family_Size"]+1)

# SET = ["Title", "Deck", "Family", "Age*Class", "Fare_per_person", "Sex", "Embarked"]
#--------------------------------------------------------------------------------------------#


############### Get some statistics #######################

############### Those who survived versus those who didn't#

survived_set = train_set.loc[train_set["Survived"] == 1, :]
dead_set = train_set.loc[train_set["Survived"] == 0, :]

survived_females = survived_set.loc[survived_set["Sex"] == 'female',:]
survived_males =   survived_set.loc[survived_set["Sex"] == 'male',:]

dead_females = dead_set.loc[dead_set["Sex"] == 'female',:]
dead_males = dead_set.loc[dead_set["Sex"] == 'male',:]

smn = survived_males.shape[0]
sfn = survived_females.shape[0]
dmn = dead_males.shape[0]
dfn = dead_females.shape[0]

import matplotlib.pyplot as pp

fig, axes = pp.subplots(nrows = 1, ncols = 2)
pp.axes(axes[0])
survived = pp.bar([0.5, 3.5], [smn, sfn], width = 1, color='#3BB200')
died =  pp.bar ([1.5, 4.5], [dmn, dfn], width = 1, color= 'red')

pp.xticks([1,4], ('Male', 'Female'))
pp.ylabel('Number of passengers')
pp.legend((survived,died), ('Survived', 'Died'), loc = 0, fontsize = 'medium')
#fractions
pp.axes(axes[1])
survived_pct = pp.bar([0.5, 3.5], [smn/(smn+dmn), sfn/(sfn+dfn)], width =1, color='#3BB200')
dead_pct = pp.bar([1.5, 4.5], [dmn/(smn+dmn), dfn/(sfn+dfn)], width = 1, color='red')

pp.xticks([1,4], ('Male', 'Female'))
pp.ylabel('Franctions of passengers')
pp.legend((survived_pct,dead_pct), ('Survived', 'Died'), fontsize = 'medium')
fig.suptitle('Sex vs. survival', fontsize = 'x-large', y=1.03)
pp.tight_layout()
pp.show()


##################### Now Find correlation between survival and age######################
def checkNans(arr, arr2=None):
    mask_nan = pd.isnull(arr) # using pandas isnull to also operate
                              # on string fields
    if mask_nan.sum()>0:
        any_nan = True
    else:
        any_nan = False
    n_nan = mask_nan.sum()
    
    masked_arr = arr[~mask_nan]
    if arr2 is not None:
        masked_arr2 = arr2[~mask_nan]
    else: 
        masked_arr2 = None

    return any_nan, masked_arr, masked_arr2, n_nan, mask_nan

survived_age = checkNans(survived_set['Age'])[1]
dead_age = checkNans(dead_set['Age'])[1]

stacked = np.hstack((survived_age, dead_age))
bins = np.histogram(stacked, bins = 16, range = (0,stacked.max()))[1]

survived = pp.hist(survived_age, bins, normed=1, facecolor='green', alpha=0.5)
dead = pp.hist(dead_age, bins, normed=1, facecolor='red', alpha=0.5)

import matplotlib.patches as mpatches
survived_handle = mpatches.Patch(facecolor='green', alpha =0.5, label = 'Survived', edgecolor= 'black')
dead_handle = mpatches.Patch(facecolor = 'red' , alpha =0.6, label = 'Dead', edgecolor = 'black')

pp.legend((survived_handle,dead_handle), ('Survived', 'Died'), loc = 0, fontsize = 'medium')

pp.title('Age vs. survival', fontsize = 'x-large', y=1.02)
pp.xlabel('Age [years]')
pp.ylabel('Fraction')
pp.xlim([0,stacked.max()])
pp.tight_layout()
pp.show()

sma = checkNans(survived_males['Age'])[1]
dma = checkNans(dead_males['Age'])[1]
sfa= checkNans(survived_females['Age'])[1]
dfa = checkNans(dead_females['Age'])[1]

fig, axes = pp.subplots(nrows = 1, ncols = 2, figsize=(8,4), sharey= True)
pp.axes(axes[0])

survived_male = pp.hist(sma, bins, normed=1, facecolor = 'green', alpha=0.5)
dead_male = pp.hist(dma, bins, normed = 1, facecolor = 'red', alpha=0.5)

pp.legend((survived_handle,dead_handle), ('Survived', 'Died'), loc = 0, fontsize ='medium')

pp.title('Male')
pp.xlabel('Age [years]')
pp.ylabel('Fraction')
pp.xlim([0,stacked.max()]) # Using the same range as in the previous plot
pp.tight_layout()

pp.axes(axes[1])
survived_female = pp.hist(sfa, bins, normed=1, facecolor = 'green', alpha=0.5)
dead_female = pp.hist(dfa, bins, normed = 1, facecolor = 'red', alpha=0.5)

pp.legend((survived_handle,dead_handle), ('Survived', 'Died'),  fontsize ='medium')

pp.title('Female')
pp.xlabel('Age [years]')
pp.ylabel('Fraction')
pp.xlim([0,stacked.max()]) # Using the same range as in the previous plot
pp.tight_layout()

fig.suptitle('Age vs. survival', fontsize = 'x-large', y=1.02)
pp.show()

# CHECK CLASS NOW#

survived_c1 = survived_set['Survived'].loc[survived_set['Pclass']==1]
dead_c1 = dead_set['Survived'].loc[dead_set['Pclass']==1]

survived_c2 = survived_set['Survived'].loc[survived_set['Pclass']==2]
dead_c2 = dead_set['Survived'].loc[dead_set['Pclass']==2]

survived_c3 = survived_set['Survived'].loc[survived_set['Pclass']==3]
dead_c3 = dead_set['Survived'].loc[dead_set['Pclass']==3]

s1n = survived_c1.shape[0]
d1n = dead_c1.shape[0]
s2n = survived_c2.shape[0]
d2n = dead_c2.shape[0]
s3n = survived_c3.shape[0]
d3n = dead_c3.shape[0]

fig, axes = pp.subplots(nrows = 1, ncols = 2)

pp.axes(axes[0])
survived = pp.bar([0.5,3.5,6.5], [s1n,s2n,s3n], color ="green", width=1)
dead = pp.bar([1.5,4.5,7.5], [d1n,d2n,d3n], color ="red", width=1)

pp.xticks([1.5,4.5,7.5],('1st Class', '2nd Class', '3rd Class'))
pp.ylabel('No of passengers')
pp.legend((survived, died), ('Survived', 'Died'), loc=0, 
          fontsize = 'medium')

pp.axes(axes[1])
survived = pp.bar([0.5,3.5,6.5], [s1n/(s1n+d1n),s2n/(s2n+d2n),s3n/(s3n+d3n)], color ="green", width=1)
dead = pp.bar([1.5,4.5,7.5], [d1n/(s1n+d1n),d2n/(s2n+d2n),d3n/(s3n+d3n)], color ="red", width=1)
pp.xticks([1.5,4.5,7.5],('1st Class', '2nd Class', '3rd Class'))
pp.ylabel('Fraction')
pp.legend((survived, died), ('Survived', 'Died'), loc='upper left',
          fontsize = 'medium')
fig.suptitle('Ticket class vs. survival', fontsize = 'x-large', y=1.03)
pp.tight_layout()
pp.show()

"""

#create a submission file for kaggle
predictiondf = pd.DataFrame(testdf['PassengerId'])
predictiondf['Survived']=[0 for x in range(len(testdf))]
predictiondf.to_csv('C:/Documents and Settings/DIGIT/My Documents/Google Drive/Blogs/triangleinequality/Titanic/prediction.csv',
              index=False)
	return [traindf, testdf, data_type_dict]
	"""