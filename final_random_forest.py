# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 21:06:51 2018

@author: Andrew
"""

# -*- coding: utf-8 -*-
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import time

week = 25
file  = 'fpl5.csv'

#dataset.sort_values(['name','gw'], inplace=True)
trees = 25
# prepare the data 
X,y,x_try_one, dataset_reduce, gameweek, x_next_game, dataset = Main_run(file,week)
#run the model
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = trees)
regressor.fit(X,y)
#option to run SVM model
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

#applying grid search to find the best model
from sklearn.model_selection import GridSearchCV
parameters = [{'C':[1], 'kernel':['linear']},{'C':[1],'kernel':['rbf'], 'gamma':[0.5] }]
grid_search = GridSearchCV(estimator = regressor, param_grid = parameters)
grid_search = grid_search.fit(X, y)
best_parameters = grid_search.best_params_

#use model to maximise best team
total, Price = Main_price(dataset,regressor,x_next_game, x_try_one,gameweek)

#create all combinations of players
FF_str, FF_def, FF_mid,FF_str2, FF_def2, FF_mid2,FF_str3, FF_def3, FF_mid3,keepers_price, defenders_price, mid_price, striker_price = create_formations(Price)
#calculate the points for all combinations
defo = calculateOptions_single(defenders_price, FF_def)
mido = calculateOptions_single(mid_price, FF_mid)
striko =  calculateOptions_single(striker_price, FF_str)
defo2 = calculateOptions_single(defenders_price, FF_def2)
mido2 = calculateOptions_single(mid_price, FF_mid2)
striko2 =  calculateOptions_single(striker_price, FF_str2)
defo3 = calculateOptions_single(defenders_price, FF_def3)
mido3 = calculateOptions_single(mid_price, FF_mid3)
striko3 =  calculateOptions_single(striker_price, FF_str3)

#find the cheapest players to have as subs
dataset_pure = pd.read_csv(file)
dataset_pure = dataset_pure[dataset_pure.gw == week]
dataset_pure = dataset_pure[['name','pos','current_price']]
defen = dataset_pure.loc[(dataset_pure['pos'] == 'DEF')]
mid = dataset_pure.loc[(dataset_pure['pos'] == 'MID')]
strik = dataset_pure.loc[(dataset_pure['pos'] == 'FOR')]
keep = dataset_pure.loc[(dataset_pure['pos'] == 'GK')]

defen = cheapest3(defen)
mid= cheapest3(mid)
strik = cheapest3(strik)
keep = cheapest3(keep)

#find the best formation. 
subs, budget = maximum_r(2,0,1,defen,mid,strik,keep)
result = results(defo,mido3,striko2,keepers_price,budget)# 3-5-2  (2def 1str)

subs, budget = maximum_r(2,1,0,defen,mid,strik,keep)
result1 = results(defo,mido2,striko3,keepers_price,budget)# 3-4-3 (2def 1mid)

subs, budget = maximum_r(1,1,1,defen,mid,strik,keep)
result2 = results(defo2,mido2,striko2,keepers_price,budget)#4-4-2 (1def 1mid 1stri)

subs, budget = maximum_r(1,0,2,defen,mid,strik,keep)
result3 = results(defo2,mido3,striko,keepers_price,budget)#4-5-1 (1def 2strik)

subs, budget = maximum_r(1,2,0,defen,mid,strik,keep)
result4 = results(defo2,mido,striko3,keepers_price,budget)#4-3-3 (1def, 2mid)

subs, budget = maximum_r(0,2,1,defen,mid,strik,keep)
result5 = results(defo3,mido,striko2,keepers_price,budget)#5-3-2 (2mid, 1str)

subs, budget = maximum_r(0,1,2,defen,mid,strik,keep)
result6 = results(defo3,mido2,striko,keepers_price,budget)#5-4-1 (1mid,2str)


print(result['points'][0])
print(result1['points'][0])
#print(result2['points'][0])
print(result3['points'][0])
print(result4['points'][0])
print(result5['points'][0])
print(result6['points'][0])

print(result5['formation'][0])
t3 = time.time()
total_time1 = t3-t2
print(total_time1)


def Main_run(file,week):    
    players=injury()
    dataset = initial_clean(file, players)
    dataset5, dataset_reduce = adding_previous_weeks(dataset)
    x_try, x_try_one, gameweek = input_variable(week,dataset5)
    y = dataset_reduce.iloc[:, [9]].values
    X, x_next_game = Encode_and_label(dataset_reduce, x_try)
    y = np.ravel(y)
    return (X,y,x_try_one, dataset_reduce, gameweek, x_next_game, dataset)
        
    
def Main_price(dataset,regressor,x_next_game, x_try_one, gameweek):        
    Price = Price_make(dataset,regressor,x_next_game, x_try_one)
        
    #actual_points = gameweek[['name','points']]
    gameweek.reset_index(drop=True, inplace=True)
    gameweek = gameweek[['id','points']]
    gameweek = gameweek.rename(columns={'points': 'actual points'})
    Price = pd.merge(Price, gameweek, on='id', how='outer')
    
    Price['difference'] = (abs(Price['point'].sub(Price['actual points'],axis=0)))**2
    Total=(Price['difference'].sum())/len(Price)
    Price.sort_values(by=['point'], ascending=False, inplace=True)
    Price.reset_index(drop=True, inplace=True)
    Price = Price.dropna(subset =['point'])
    
    return (Total,Price)
    
    
#what accuarcy measure is good....

#now i need to work out how i can nicely say how good it is.. 
#do something that picks each week works out all the players and then compares accurary.


def injury():
    from lxml import html
    import requests
    players =[]
    
    webpage = ('http://www.physioroom.com/news/english_premier_league/epl_injury_table.php')
    page = requests.get(webpage)
    tree = html.fromstring(page.content)
    
    for var in range(1, 300):
        player = tree.xpath('//*[@id="one-col"]/div[2]/div[2]/table/tbody/tr[%s]/td[1]/a/text()' % var)
        if(len(player)== 1):
            player = player[0].split(" ",1)[1]
            player = player.lstrip()
            player = player.split(" ")[1]
        
            #player = player[0].split(" ",1)[1]
        #player = str(player)
        #player = player.split(" ",1)[1] 
            players.append(player)
    return players

def Price_make(dataset,regressor,x_next_game, x_try_one):
    y_next_week_points = regressor.predict(x_next_game)
    x_try_one['point'] = y_next_week_points
    point = pd.DataFrame(x_try_one)
    total_point= point[['id','point']]
    
    dataset_t = dataset.dropna(subset =['mins'])
    dataset_t2 = dataset_t.drop_duplicates(subset='id', keep='last', inplace=False)
    Price = dataset_t2[['price','id','pos','name','team']]

   # Price = pd.DataFrame(Price, columns = ['price','id','position','name'])
    Price = pd.merge(Price, total_point, on='id', how='outer')

    Price.sort_values(by=['point'], ascending=False, inplace=True)
    Price.reset_index(drop=True,inplace=True)
    #Price = Price[Price.point > 2]

    return Price


def initial_clean(file, players):
    dataset = pd.read_csv(file)
    dataset.sort_values(['name','gw'], inplace=True)
    dataset = dataset[~dataset.name.isin(players)]
    dataset.reset_index(drop=True, inplace=True)
    return dataset

#Adding historic previous weeks data to the same 'line'.
def adding_previous_weeks(dataset):

    dataset3 = dataset.drop(dataset.index[:1])
    dataset3.reset_index(drop=True, inplace=True)
    dataset3['previous'] = dataset['points']
    dataset3['goals p'] = dataset['goals']
    dataset3['assists p'] = dataset['assists']
    dataset3['key_passes_p'] = dataset['key_passes']
    
    dataset4 = dataset3.drop(dataset.index[:1])
    dataset4.reset_index(drop=True, inplace=True)
    dataset4['previous minus one'] = dataset3['previous']
    dataset4['goals p1'] = dataset3['goals p']
    dataset4['assists p1'] = dataset3['assists p']
    dataset4['key_passes_p1'] = dataset3['key_passes_p']
    
    dataset5 = dataset4.drop(dataset.index[:1])
    dataset5.reset_index(drop=True, inplace=True)
    dataset5['previous minus two'] = dataset4['previous minus one']
    dataset5['goals p2'] = dataset4['goals p1']
    dataset5['assists p2'] = dataset4['assists p1']
    dataset5['key_passes_p2'] = dataset4['key_passes_p1']
    
    #remove all week 1,2,3. above approach overlaps players. 
    dataset5 = dataset5[dataset5.gw != 1]
    dataset5 = dataset5[dataset5.gw != 2]
    dataset5 = dataset5[dataset5.gw != 3]
    
    dataset5.reset_index(drop=True, inplace=True)
    
    dataset_reduce = dataset5[['gw','home','oppo_team','name','previous','previous minus one','previous minus two','id','mins','points']]
    dataset_reduce.dropna(inplace = True, subset =['mins'])
    dataset_reduce.dropna(inplace = True, subset =['previous','previous minus one','previous minus two'])
    dataset_reduce.drop(['mins'],axis=1)
    
    return (dataset5, dataset_reduce)

#creating the inputdata for modelling the next week. should be called x_something. 
def input_variable(week,dataset):
    gameweek = dataset[dataset.gw == week]
    pre_input = gameweek[['home','oppo_team','name','previous','previous minus one','previous minus two','id']]
    gameweek = gameweek[['home','oppo_team','name','previous','previous minus one','previous minus two','id','points']]
    x_try_one = pre_input.dropna(subset =['previous','previous minus one','previous minus two'])
    x_try  = x_try_one.drop(['id'],axis=1)
    return (x_try, x_try_one, gameweek)


def Encode_and_label(dataset_reduce, x_try):
    X = dataset_reduce.iloc[:, [1,2,3,4,5,6]].values
    x_next_game = x_try.iloc[:,:].values
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    labelencoder_X1 = LabelEncoder()
    labelencoder_X2 = LabelEncoder()
    labelencoder_X3 = LabelEncoder()
    X[:, 0] = labelencoder_X1.fit_transform(X[:, 0])
    X[:, 1] = labelencoder_X2.fit_transform(X[:, 1])
    X[:, 2] = labelencoder_X3.fit_transform(X[:, 2])
    x_next_game[:, 0] = labelencoder_X1.transform(x_next_game[:, 0])
    x_next_game[:, 1] = labelencoder_X2.transform(x_next_game[:, 1])
    x_next_game[:, 2] = labelencoder_X3.transform(x_next_game[:, 2])
    
    onehotencoder = OneHotEncoder(categorical_features = [0,1,2])
    X = onehotencoder.fit_transform(X).toarray()
    x_next_game = onehotencoder.transform(x_next_game).toarray()
    return(X,x_next_game)

    #we need to remove a couple of columns to avoid the dummy variable trap.. but which ones.
def create_formations(Price):  
    defenders_price = Price.loc[(Price['pos'] == 'DEF')]
    defenders_price.reset_index(drop=True,inplace=True)
    defenders_price = defenders_price[0:15]
    mid_price = Price.loc[(Price['pos'] == 'MID')]
    mid_price.reset_index(drop=True,inplace=True)
    mid_price = mid_price[0:15]
    striker_price = Price.loc[(Price['pos'] =='FOR')]
    striker_price.reset_index(drop=True,inplace=True)
    striker_price = striker_price[0:15]
    keepers = Price.loc[(Price['pos'] == 'GK')]      
    keepers.reset_index(drop=True,inplace=True)
    keepers = keepers[0:15]

    FF_def= []
    FF_mid= []
    FF_str= []
    FF_def2= []
    FF_mid2= []
    FF_str2= []
    FF_def3= []
    FF_mid3= []
    FF_str3= []

    for defend in itertools.combinations(defenders_price['id'], 3):
        FF_def.append(defend)
    for defend in itertools.combinations(defenders_price['id'], 4):
        FF_def2.append(defend)
    for defend in itertools.combinations(defenders_price['id'], 5):
        FF_def3.append(defend)
        
    for mid in itertools.combinations(mid_price['id'], 3):
        FF_mid.append(mid)
    for mid in itertools.combinations(mid_price['id'], 4):
        FF_mid2.append(mid)
    for mid in itertools.combinations(mid_price['id'], 5):
        FF_mid3.append(mid)
    
    for str in itertools.combinations(striker_price['id'], 1):
        FF_str.append(str)
    for str in itertools.combinations(striker_price['id'], 2):
        FF_str2.append(str)
    for str in itertools.combinations(striker_price['id'], 3):
        FF_str3.append(str)
    return (FF_str, FF_def, FF_mid,FF_str2, FF_def2, FF_mid2,FF_str3, FF_def3, FF_mid3, keepers,defenders_price,mid_price,striker_price)

def calculateOptions_single(pri, formation):

    df3 = []
    for form in formation:
        tc=pd.DataFrame()
        for i in range(0,len(form)):
            vv=(pri.loc[pri['id'] ==form[i]])
            tc= tc.append(vv) 
        tc.reset_index(drop=True,inplace=True)
        total_cost = 0
        total_points = 0
        teams=[]
        names = []
        for player in range(0,len(form)):
            total_cost = total_cost + tc['price'][player]
            total_points = total_points + tc['point'][player]
            names.append(tc['name'][player])
            teams.append(tc['team'][player])
        df2 = [total_cost,total_points, form, names,teams]
        df3.append(df2)
    options = pd.DataFrame(df3,columns = ['price','points','formation','names','teams'] )
    options.sort_values(by=['points'], ascending=False, inplace=True) 
    options.reset_index(drop=True,inplace=True)
    options= options[0:100]     
    return options   

def results(defo,mido,striko,keepers_price, budget):
    result = []
    defo = defo[0:10]
    mido = mido[0:10]
    striko = striko[0:10]
    for index, row in defo.iterrows():
        for index, row2 in mido.iterrows():
            for index, row3 in striko.iterrows():
                for index, row4 in keepers_price.iterrows():
                    total_cost = row['price'] + row2['price']+row3['price']+ row4['price'] 
                    total_points = row['points'] + row2['points'] + row3['points'] + row4['point']
                    formation = [row['names'],row2['names'],row3['names'],row4['name']]
                    teams = [row['teams'],row2['teams'],row3['teams'],row4['team']]
                    output = [total_cost,total_points,formation,teams]
                    
                
                    result.append(output)
    
    result = pd.DataFrame(result, columns=['cost','points','formation','teams'])
    
    
    # limit depends on formation... 
    result = result[result.cost < budget]
    result = result.sort_values(by=['points'], ascending=False)
    result.reset_index(drop=True, inplace=True)
    
    remove =[]
    for ind in range(len(result)):
        team_t = result['teams'][ind] 
        teamo =[]
        for one in range(len(team_t[0])):
            teamo.append(team_t[0][one])   
        for one in range(len(team_t[1])):
            teamo.append(team_t[1][one])
        for one in range(len(team_t[2])):
            teamo.append(team_t[2][one])
        teamo.append(team_t[3])
        unique = set(teamo)
        
        for id in unique:
        #print(teamo.count(id))
        
            if (teamo.count(id) > 3):
            #print (teamo.count(id))
                remove.append(ind)
                

    result = result.drop(remove)    
    result.reset_index(drop=True, inplace=True)
    return result

#get cheapest 3
def cheapest3(defen):
    defen.sort_values(by=['current_price'], ascending=True, inplace=True) 
    defen.reset_index(drop=True,inplace=True)
    defen = defen[0:3]
    return defen


#calculate the maximum total price for each formation

def maximum_r(a,b,c,defen,mid,strik, keep):
    subs_cost = keep['current_price'][0]
    subs = []
    if(a==0):
        subs_cost = 0
    else:
        if(a ==1):
            subs_cost = defen['current_price'][a-1]
            subs.append(defen['name'][a-1])
        else:
            subs_cost = defen['current_price'][a-1] + defen['current_price'][a-2]
            subs.append(defen['name'][a-1])
            subs.append(defen['name'][a-2])
    
    if(b==0):
        subs_cost = subs_cost + 0
    else:
        if(b ==1):
            subs_cost = subs_cost + mid['current_price'][b-1]
            subs.append(mid['name'][b-1])
        else:
            subs_cost = subs_cost + mid['current_price'][b-1] + mid['current_price'][b-2]  
            subs.append(mid['name'][b-1])
            subs.append(mid['name'][b-2])
            
    if(c==0):
            subs_cost = subs_cost + 0
    else:
        if(c ==1):
            subs_cost = subs_cost + strik['current_price'][c-1]
            subs.append(strik['name'][c-1])
        else:
            subs_cost = subs_cost + strik['current_price'][c] + strik['current_price'][c-2] 
            subs.append(strik['name'][c-1])
            subs.append(strik['name'][c-2])
    
    budget = 1000-subs_cost        
    return(subs, budget)
       
            
            


























#save the x_next_game to send to the tensorflow model

X_save = pd.DataFrame(X)
y_add = pd.DataFrame(y)
X_save['point']=y_add[0]
X_save.to_csv(path_or_buf=r'C:\Users\Andrew\Documents\3. Fantasy Football\sparkin.csv')

input_frame_save = pd.DataFrame(x_next_game)
input_frame_save.to_csv(path_or_buf=r'C:\Users\Andrew\PycharmProjects\tensorflow\Ex_Files_TensorFlow\Ex_Files_TensorFlow\Fantasy one\next_game.csv')


querky = pd.read_csv(r'C:\Users\Andrew\PycharmProjects\tensorflow\Ex_Files_TensorFlow\Ex_Files_TensorFlow\Fantasy one\next_game.csv')



#save to file the X  and y so i can use them with tensorflow. 
frame_save = pd.DataFrame(X_train)
frame_save.to_csv(path_or_buf=r'C:\Users\Andrew\PycharmProjects\tensorflow\Ex_Files_TensorFlow\Ex_Files_TensorFlow\Fantasy one\input_train.csv')
frame_save_y = pd.DataFrame(y_train)
frame_save_y.to_csv(path_or_buf=r'C:\Users\Andrew\PycharmProjects\tensorflow\Ex_Files_TensorFlow\Ex_Files_TensorFlow\Fantasy one\output_train.csv')

frame_save = pd.DataFrame(X_test)
frame_save.to_csv(path_or_buf=r'C:\Users\Andrew\PycharmProjects\tensorflow\Ex_Files_TensorFlow\Ex_Files_TensorFlow\Fantasy one\input_test.csv')
frame_save_y = pd.DataFrame(y_test)
frame_save_y.to_csv(path_or_buf=r'C:\Users\Andrew\PycharmProjects\tensorflow\Ex_Files_TensorFlow\Ex_Files_TensorFlow\Fantasy one\output_test.csv')


predicted_test = regressor.predict(X_test)

from sklearn.metrics import r2_score
from scipy.stats import spearmanr, pearsonr

basic_score = regressor.score(X_test,y_test)
test_score = r2_score(y_test, predicted_test)
spearman = spearmanr(y_test, predicted_test)
#pearson = pearsonr(y_test, predicted_test)

print(basic_score)
print(test_score)
print(spearman)
#print(pearson)

#either tensorflow or randomforest
y_next_week_points = pd.read_csv("tensorflow.csv")

#for tensr
y_try_one['point'] = y_next_week_points['0']
point = pd.DataFrame(y_try_one)
total_point= point[['id','point']]

