import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
import os
import time

from IPython.display import clear_output
from tensorflow import keras
from keras.layers.core import Dense
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

def refresh():
  clear_output()
  os.system('cls||clear')
  time.sleep(0.5)   

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def Read_File( filename , Daughter_Data = False , Unit_Of_Time = 'Seconds' ):
  
  DF = pd.read_csv(filepath_or_buffer = filename)

  Isotope_List = list(DF.pop('Isotope'))

  if Daughter_Data == False:
    DF = Set_Unit_Of_Time( DF , Unit_Of_Time = Unit_Of_Time)  

    try:

      if Daughter_Data == False:
          DF = DF.drop(columns=['Daughter_1','Daughter_1_Half_Life({:})'.format(Unit_Of_Time),'Type_of _Decay_2'])

    except:
      print("ERROR")
      pass

  else:
    DF = Set_Unit_Of_Time( DF , Unit_Of_Time = Unit_Of_Time) 
    DF = Set_Unit_Of_Time( DF , Unit_Of_Time = Unit_Of_Time , Csv_Col='Daughter_1_Half_Life')

  return DF , Isotope_List 


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#This Function adds gaussian noise to a set of data when creatiing the digital twin.

#'Data' is the input data that the noise is going to be apply to , 'mu' is the mean , and 'std' is standard deviation.  


def gaussian_noise( Data , mu = 0.0 , std = 0.01 ):

  noise = np.random.normal( mu , std, size = Data.shape)

  Noisy_Data = Data + noise

  if Noisy_Data <=0:

    return gaussian_noise( Data , mu = 0.0 , std = std)

  return Noisy_Data 


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


#This function adds guassian noise to the data which is used to create a digital twin.



def Exp_form( X0 , Parent_decay_constant , Daughter_decay_constant = None , t = None , Daughter_Data = False ):
  
  if Daughter_Data == False:

    X = X0 * np.e**(-Parent_decay_constant*t) 
    
  elif Daughter_Data == True:

    X = X0 * ( Parent_decay_constant / (Daughter_decay_constant - Parent_decay_constant )  * ( (np.e**(-Parent_decay_constant * t) )  - (np.e**(-Daughter_decay_constant*t) ) ) )  

  return  X 




#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#



def Set_Unit_Of_Time( DF , Unit_Of_Time = None , Csv_Col='Half_Life'):

  if Csv_Col == 'Half_Life':
      Original_Time = list(DF.columns)[0]
  else:
      Original_Time = list(DF.columns)[3]


  if Original_Time == '{:}(Seconds)'.format(Csv_Col):


      if Unit_Of_Time == 'Minutes':

          DF[Original_Time] = DF[Original_Time].map(lambda x : x/60 )

          DF.rename(columns={Original_Time :'{:}(Minutes)'.format(Csv_Col)} ,   inplace = True)

      elif Unit_Of_Time == 'Hours':

          DF[Original_Time] = DF[Original_Time].map(lambda x : x/3600 )

          DF.rename(columns={Original_Time :'{:}(Hours)'.format(Csv_Col)}   ,   inplace = True)

      elif Unit_Of_Time == 'Days':

          DF[Original_Time] = DF[Original_Time].map(lambda x : x/(24*3600))

          DF.rename(columns={Original_Time :'{:}(Days)'.format(Csv_Col)}    ,   inplace = True)


  if Original_Time == '{:}(Minutes)'.format(Csv_Col):

      if Unit_Of_Time == 'Seconds':

          DF[Original_Time] = DF[Original_Time].map(lambda x : x*60 )

          DF.rename(columns={Original_Time :'{:}(Seconds)'.format(Csv_Col)} ,   inplace = True)


      elif Unit_Of_Time == 'Hours':

          DF[Original_Time] = DF[Original_Time].map(lambda x : x/60 )

          DF.rename(columns={Original_Time :'{:}(Minutes)'.format(Csv_Col)} ,   inplace = True)

      elif Unit_Of_Time == 'Days':

          DF[Original_Time] = DF[Original_Time].map(lambda x : x/(24*60))

          DF.rename(columns={Original_Time :'{:}(Days)'.format(Csv_Col)}    ,   inplace = True)


      else:   pass


  if Original_Time == '{:}(Hours)'.format(Csv_Col):

      if Unit_Of_Time == 'Seconds':

          DF[Original_Time] = DF[Original_Time].map(lambda x : x*3600)

          DF.rename(columns={Original_Time :'{:}(Seconds)'.format(Csv_Col)} ,   inplace = True)


      elif Unit_Of_Time == 'Minutes':

          DF[Original_Time] = DF[Original_Time].map(lambda x : x*60)

          DF.rename(columns={Original_Time :'{:}(Minutes)'.format(Csv_Col)} ,   inplace = True)


      elif Unit_Of_Time == 'Days':

          DF[Original_Time] = DF[Original_Time].map(lambda x : x/24)

          DF.rename(columns={Original_Time :'{:}(Days)'.format(Csv_Col)}    ,   inplace = True)


      else:   pass


  if Original_Time == '{:}(Days)'.format(Csv_Col):

      if Unit_Of_Time == 'Seconds':

          DF[Original_Time] = DF[Original_Time].map(lambda x : x/(24*3600) )

          DF.rename(columns={Original_Time :'{:}(Seconds)'.format(Csv_Col)} ,   inplace = True)

      elif Unit_Of_Time == 'Minutes':

          DF[Original_Time] = DF[Original_Time].map(lambda x : x/(24*60) )

          DF.rename(columns={Original_Time :'{:}(Minutes)'.format(Csv_Col)} ,   inplace = True)

      elif Unit_Of_Time == 'Hours':

          DF[Original_Time] = DF[Original_Time].map(lambda x : x/24 )

          DF.rename(columns={Original_Time :'{:}(Minutes)'.format(Csv_Col)} ,   inplace = True)

      else:   pass
    

  return DF


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


def Create_Data_Set( DF , std =0.01 , Num_Of_Replicates = 5 , Unit_Of_Time = 'Seconds'):
  time_list = []
  k= 0
  while k<1:
    time_list.append(k)
    k+=0.001
  [ time_list.append(i) for i in range(1,10000)]

  

  Original_Type_Of_Decay= DF.pop('Type_of _Decay_1')

  Type_Of_Decay =  np.sort ( Original_Type_Of_Decay.unique().tolist() )

  Type_Of_Decay = { i:index for i,index in zip( Type_Of_Decay , range(len(Type_Of_Decay)) )}


  DF = np.array(DF)



  Decay_Nest = []

  for  i in range( len(DF) ):

      Decay_Nest.append([gaussian_noise(DF[i] , mu = 0.0 , std = std ) for iteraton in range(  Num_Of_Replicates )])




  Decay_const = list(map(lambda x : (np.log(2)/x) , Decay_Nest))


  Ap_list={'N'           : [],
            't'           : [],
            'Decay_Type'  : [],
            'Isotope'     : []}


  N0 = 1

  for t,zipper in zip(time_list,range(len(time_list))):

      for isotope , index in zip(Decay_const,range(len(Decay_const))):


          for isotope_Decay_constant in isotope :

            
              N=Exp_form( X0 = N0 ,Parent_decay_constant = isotope_Decay_constant , t = t )
            
              Ap_list['N'].append( N[0] )
              Ap_list['t'].append(t)
              Ap_list['Decay_Type'].append( Type_Of_Decay[str(Original_Type_Of_Decay[index])] )    
              Ap_list['Isotope'].append(index)


      print('{} / {}'.format(zipper, len(time_list)))


  DF = pd.DataFrame(Ap_list)



  return DF

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def Create_Data_Set_2( DF , std =0.01 , Num_Of_Replicates = 10 , Unit_Of_Time = 'Seconds'):

  time_list = [ i for i in range(0,31557600,500)]



  Original_Type_Of_Decay_1= DF.pop('Type_of _Decay_1')
  Original_Type_Of_Decay_2= DF.pop('Type_of _Decay_2')


  Decay_Combinations = []

  for  Decay_1 , Decay_2 in zip(Original_Type_Of_Decay_1 , Original_Type_Of_Decay_2 ):

    combination = [Decay_1,Decay_2]

    if combination not in Decay_Combinations:

      Decay_Combinations.append(combination)

  Type_Of_Decay = { str(i) :index for i , index in zip( Decay_Combinations , range(len( Decay_Combinations )) )}


  

  Parent_Half_Life = np.array( pd.DataFrame( DF['Half_Life({:})'.format(Unit_Of_Time)] )  )


  Daughter_Half_Life = np.array( pd.DataFrame( DF['Daughter_1_Half_Life({:})'.format(Unit_Of_Time)] )   )

  Parent_Decay_Nest = []
  
  for  i in range( len(Parent_Half_Life) ):

      Parent_Decay_Nest.append( [ gaussian_noise(Parent_Half_Life[i] , mu = 0.0 , std = std )  for iteraton in range(  Num_Of_Replicates ) ] )

  Daughter_Decay_Nest = []

  for  i in range( len(Daughter_Half_Life) ):

      Daughter_Decay_Nest.append( [ gaussian_noise(Daughter_Half_Life[i] , mu = 0.0 , std = std ) for iteraton in range(  Num_Of_Replicates ) ])


  Parent_Decay_Const = list(map(lambda x : (np.log(2)/x) , Parent_Half_Life))
  Daughter_Decay_Const = list(map(lambda x : (np.log(2)/x) , Daughter_Half_Life))


  Ap_list={ 'N'           : [],
            't'           : [],
            'Decay_Type'  : [],
            'Isotope'     : []}

  

  N0 = 1

  for t,zipper in zip(time_list,range(len(time_list))):

      for Parent_isotope,  Daughter_Isotope , index in zip( Parent_Decay_Const, Daughter_Decay_Const ,  range(len(Parent_Decay_Const))  ):


          for P_isotope_Decay_constant , D_isotope_Decay_constant in zip(Parent_isotope ,Daughter_Isotope)  :

              if D_isotope_Decay_constant == np.inf:

                N = Exp_form( N0 , P_isotope_Decay_constant , t = t )
              else:

                N =  Exp_form( N0 , P_isotope_Decay_constant , t = t ) + Exp_form( N0 , Parent_decay_constant = P_isotope_Decay_constant  , Daughter_decay_constant = D_isotope_Decay_constant , t = t , Daughter_Data = True )  # Daughter Decay
                                                                                                                                                              
            
              Ap_list['N'].append( N )
              Ap_list['t'].append(t)
              Ap_list['Decay_Type'].append( Type_Of_Decay["{:}".format( [ Original_Type_Of_Decay_1[index] , Original_Type_Of_Decay_2[index] ] ) ] )    
              Ap_list['Isotope'].append(index)


      print('{} / {}'.format(zipper, len(time_list)))



  DF = pd.DataFrame(Ap_list)



  return DF




#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#



def Plot_Data_Frame(filename , Unit_Of_Time = 'Seconds' ,Daughter_Data = False):
  warnings.filterwarnings('ignore')
  print("Plotting Data please wait ...")

  DF , Isotope_List = Read_File( filename , Daughter_Data = Daughter_Data , Unit_Of_Time = Unit_Of_Time)
  
  Time_Dict = { i : [ ] for i in range(0,(int(input("Max:")))+1 , int(input("skips:"))) }
  
  N0=1

  if Daughter_Data == False:


      DF = pd.DataFrame(DF['Half_Life({})'.format( Unit_Of_Time )])

      DF['Half_Life({})'.format( Unit_Of_Time )]= DF['Half_Life({})'.format( Unit_Of_Time )].map(lambda x : np.log(2)/x )


      for P_D_C in list(DF.values):
      
          for t in Time_Dict:

              Time_Dict[t].append(Exp_form( X0 = N0 , Parent_decay_constant=P_D_C  , t = t ))


  if Daughter_Data == True:


      Parent_Half_Life = np.array( pd.DataFrame( DF['Half_Life({:})'.format(Unit_Of_Time)] )  )

      Parent_Decay_Const = list(map(lambda x : (np.log(2)/x) , Parent_Half_Life))


      Daughter_Half_Life = np.array( pd.DataFrame( DF['Daughter_1_Half_Life({:})'.format(Unit_Of_Time)] )   )

      Daughter_Half_Life = list(map(lambda x : (np.log(2)/x) , Daughter_Half_Life))

      for P_D_C, D_D_C in zip( Parent_Decay_Const, Daughter_Half_Life ):
          
          if D_D_C == np.inf:
              for t in Time_Dict:
                                  
                      N = Exp_form( X0 = N0 , Parent_decay_constant= P_D_C  , t = t )
                      Time_Dict[t].append(N[0])

          else:
              for t in Time_Dict:

                      N = sum( Exp_form( X0 = N0 , Parent_decay_constant= P_D_C  , t = t ) , Exp_form( X0 = N0 , Parent_decay_constant = P_D_C , Daughter_decay_constant = D_D_C , t = t , Daughter_Data = True ) )
                      Time_Dict[t].append(N[0])



  Time_Dict = pd.DataFrame(Time_Dict)  
  

  X_values = list(Time_Dict.columns.values)

  for index, rows in Time_Dict.iterrows():
      # Create list for the current row
      plt.plot(X_values , list(Time_Dict.iloc[index]) )

  plt.xlabel(Unit_Of_Time)
  plt.show()
  plt.close('all')



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


def training_V2 ( Training_df  ,Isotope_List , New_Model=True , model = None):

  if New_Model== True: 

      train_labels = Training_df.pop('Isotope')

      scaled_DF ,train_labels = shuffle(Training_df ,train_labels )

      scaler = MinMaxScaler()

      scaled_DF[['t','Decay_Type']] = scaler.fit_transform(scaled_DF[['t','Decay_Type']])


      model = Sequential([Dense(16, input_shape=((scaled_DF.shape)[1],) ,activation='sigmoid'),
                          Dense(32, activation='sigmoid'),
                          Dense(len(Isotope_List),activation='softmax')])


      model.compile(optimizer=Adam(learning_rate=0.01), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy',])



      history  = model.fit(   scaled_DF, train_labels , 

                              validation_split=0.1 , epochs=10 ,

                              shuffle=True , verbose=2) 


      return model , history 


  elif New_Model==False:

      sx = MinMaxScaler()
      sy = MinMaxScaler()
      scaled_df = sx.fit_transform(Training_df.drop('Isotope', axis='columns' ))
      
      df_train = Training_df.pop('Isotope')

      Training_df,df_train = shuffle( Training_df, df_train )

      scaler = MinMaxScaler(feature_range=(0,1))

      scaled_df = scaler.fit_transform(Training_df)


      tf.random.set_seed(42)

      history  = model.fit(scaled_df, df_train, epochs=50 ,verbose=2)

      return model , history 


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


def Isotope_Shopping_List(Isotope_List , Shopping_List=[] ,Shopping =True):

  Isotope_Identification = { i:index for i,index in zip( Isotope_List , range(len(Isotope_List)) )}
  Sorted_Istope_List = np.sort(Isotope_List)


  while Shopping:

      refresh()

      for  isotope , index in zip(Sorted_Istope_List , range(len(Sorted_Istope_List)) )  : 

          print( '{:20s} : {:} '. format( isotope , index ) ) 

      print( '\n{:20s} : {:} '. format( "Show Isotope List" , "i" ) ) 

      

      print( '\n{:21s} : {:} '. format( "Back" , "q" ) ) 

      Item = input('\n\n{:22s} : '. format( "Option" ) )

      if Item.isdigit()==True :
          if int(Item) in range(len(Sorted_Istope_List)):
              Item = Isotope_Identification [ Sorted_Istope_List [int(Item)]]

          if Item not in Shopping_List : Shopping_List.append(Item)

          refresh()


      elif Item == 'i':
          
          while True:

              refresh()
              Categotical_Shopping_List=[]
              for isotope in Shopping_List  :  Categotical_Shopping_List.append( Isotope_List[isotope])

              print(Categotical_Shopping_List)

              print( '\n{:20s} : {:} '. format( "Empty Isotope List" , "1" ) )
              print( '\n{:20s} : {:} '. format( "Back" , "q" ) ) 

              Option = input("Option : ")

              if Option =="1":
                  Shopping_List = [] 

              if Option =="q":
                  break    

      elif Item == 'q':
          break 

  if Shopping_List == []  : return

  else : return Shopping_List

  







#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def Evaluate( Testing_df , model , Unknown_Isotope = False):

  df_test_eval = None
  scaler = MinMaxScaler()

  try:
      df_test_eval = Testing_df.pop('Isotope')
      scaled_df_test,df_test_eval = shuffle(Testing_df,df_test_eval)
  
  except:
      pass

  try:

      scaled_df_test[['t','Decay_Type']] = scaler.fit_transform(scaled_df_test[['t','Decay_Type']])

      eval_result = model.predict(x=scaled_df_test)
  
  except:
      scaled_df_test = Testing_df

      if Unknown_Isotope == True:
          scaled_df_test[['t']] = scaler.fit_transform( scaled_df_test[['t']] )
          eval_result = model.predict(x=scaled_df_test)

  rounded_predictions = np.argmax(eval_result,axis=-1)

  if Unknown_Isotope == False:

      model.evaluate(x=scaled_df_test , y=df_test_eval , verbose=1 )
      

  return eval_result  , df_test_eval 

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


def Further_Evalutaion ( eval_result, Isotope_List , df_test_eval=None , Unknown_Isotope = False , Radioactive_Shopping_List = False  , Shopping_List = None):


  if Unknown_Isotope == False:
      
      if Radioactive_Shopping_List  == False :
          Ordered_Predictions = (-eval_result).argsort()

          Most_Probable_Index_1st = []
          
          Most_Probable_Index_2nd  = []

          Most_Probable_Index_3rd = []

          for  i in Ordered_Predictions:

              Most_Probable_Index_1st.append(i[0])
              Most_Probable_Index_2nd.append(i[1])
              Most_Probable_Index_3rd.append(i[2])

          for MP1 , MP2 ,MP3, real , prob in zip( Most_Probable_Index_1st , Most_Probable_Index_2nd, Most_Probable_Index_3rd , df_test_eval , eval_result):
              
              answer = ""
              if MP1 == real :
                  answer = "Correct"
              
              else:
                  answer= "Wrong"

              print("{:10n} : {:.2f}  {:10n} : {:.2f} {:10n} : {:.2f} Answer Given: {:8s} ..... {:11n} ".format(MP1, prob[MP1], MP2 , prob[MP2] , MP3 , prob[MP3] , answer , real ))
              # print("{:20s} : {:.2f}  {:20s} : {:.2f} Answer Given: {:8s} ..... {:11s} ".format(Isotope_List[MP1], prob[MP1], Isotope_List[MP2] , prob[MP2] , answer , Isotope_List[real] ))
      
      if Unknown_Isotope == True:

          Shopping_list_prob = []
          for prediction in eval_result:

              Shopping_list_prob.append( { Isotope_List[isotope] : prediction[isotope] for isotope in Shopping_List } ) 

          for guess in Shopping_list_prob:
              
              result =""
              for Isotope , prob in zip( guess.keys() , guess.values()):

                  result = result + " {:20s}: {:.2f}".format(Isotope,prob)

              print(result)


  if Unknown_Isotope == True:

      if Radioactive_Shopping_List  == False :

          Ordered_Predictions = (-eval_result).argsort()

          Most_Probable_Index_1st = []
          
          Most_Probable_Index_2nd  = []

          Most_Probable_Index_3rd = []

          for  i in Ordered_Predictions:

              Most_Probable_Index_1st.append(i[0])
              Most_Probable_Index_2nd.append(i[1])
              Most_Probable_Index_3rd.append(i[2])

          for MP1 , MP2 ,MP3 , prob in zip( Most_Probable_Index_1st , Most_Probable_Index_2nd, Most_Probable_Index_3rd  , eval_result):
              
              # print("{:10n} : {:.2f}  {:10n} : {:.2f} {:10n} : {:.2f}  ".format(MP1, prob[MP1], MP2 , prob[MP2] , MP3 , prob[MP3]  ))
              print("{:20s} : {:.2f}  {:20s} : {:.2f}  ".format(Isotope_List[MP1], prob[MP1], Isotope_List[MP2] , prob[MP2]  ))

      if Radioactive_Shopping_List == True:
          Shopping_list_prob = []
          for prediction in eval_result:

              Shopping_list_prob.append( { Isotope_List[isotope] : prediction[isotope] for isotope in Shopping_List } ) 

          for guess in Shopping_list_prob:
              
              result =""
              for Isotope , prob in zip( guess.keys() , guess.values()):

                  result = result + " {:20s}: {:.2f} ".format(Isotope,prob)
              print(result)

          
          

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


def Unknown_Isotope ( model , filename ,Unit_Of_Time = 'Seconds' , Radioactive_Shopping_List = False , Shopping_List = None ) :

  time_list = [ i for i in range(0,10000)]

  DF , Isotope_List = Read_File( filename = filename , Unit_Of_Time = Unit_Of_Time  )


  Original_Type_Of_Decay= DF.pop('Type_of _Decay_1')

  Type_Of_Decay =  np.sort ( Original_Type_Of_Decay.unique().tolist() )

  Type_Of_Decay = { i:index for i,index in zip( Type_Of_Decay , range(len(Type_Of_Decay)) )}

  Unknown_Isotope_Dict ={ 'N'           : [],
                          't'           : [],
                          'Decay_Type'  : [] }



  decay_constant = np.log(2) / float(input("Half Life in ({:}) : ".format( Unit_Of_Time )))
  
  print(Type_Of_Decay) 
  while True:
      
      Decay_Type = input("input type of decay (word)")

      if Decay_Type in Type_Of_Decay.keys():

          break

  N0 = 1
  for t in time_list:
      N = Exp_form( N0 ,decay_constant , t = t )
      Unknown_Isotope_Dict['N'].append(N)
      Unknown_Isotope_Dict['t'].append(t)
      Unknown_Isotope_Dict['Decay_Type'].append( (Type_Of_Decay[Decay_Type])/(len(Type_Of_Decay)-1) )



      if N == 0.001* N0 :
          break

  Unknown_Isotope = pd.DataFrame(Unknown_Isotope_Dict)

  scaler = MinMaxScaler()

  # Unknown_Isotope[['t']] = scaler.fit_transform(Unknown_Isotope[['t']])

  eval_result  , df_test_eval  = Evaluate( Unknown_Isotope , model , Unknown_Isotope = True)
  if Radioactive_Shopping_List == False:
      Further_Evalutaion (    eval_result =eval_result , Isotope_List =Isotope_List , 
                              df_test_eval=df_test_eval , Unknown_Isotope = True ,
                                )
  else:  
      Further_Evalutaion (    eval_result =eval_result  , Isotope_List =Isotope_List , 
                              df_test_eval=df_test_eval , Unknown_Isotope = True ,
                              Radioactive_Shopping_List = Radioactive_Shopping_List,
                              Shopping_List = Shopping_List )

  print(Unknown_Isotope)


  return 


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def Load_DataFrame():
  DF = pd.read_pickle(input("Enter file name or path : "))

  return DF


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#



def Load_model():
  loaded_model = load_model(input("Enter file name or path : "))
  loaded_model.summary()

  return loaded_model

def Load_Example_model(filename):
  loaded_model = load_model(filename)
  loaded_model.summary()

  return loaded_model

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


def Test_func(DF , Option = None , Parent_List=None ,Unit_Of_Time = None):

  if Option == 1:

      graph_num = int(input("Input number between (1,{}) :".format(len(DF))))

      try:
          CDf=DF[graph_num-1:graph_num]
          print(CDf)
          try:
              Parent_Dc=np.log(2)/float(CDf['Half_Life({:})'.format(Unit_Of_Time)])
              Daughter_DC=np.log(2)/float(CDf['Daughter_1_Half_Life({:})'.format(Unit_Of_Time)])
          except:
              pass
          N0=5000
          Parent_X=[]
          Daughter_X=[]
          t=[]

          j=0

          while j <1000000 :  #1000000

              Parent_X.append (Exp_form( N0 , Parent_decay_constant = Parent_Dc , t = j ) )
  
              Daughter_X.append(Exp_form( X0 = N0 , Parent_decay_constant = Parent_Dc  , Daughter_decay_constant = Daughter_DC , t = j , Daughter_Data = True ) )
              
              t.append(j)

              if N0*( Parent_Dc/(Daughter_DC - Parent_Dc) )*( (np.e**(-Parent_Dc*j)) -(np.e**(-Daughter_DC*j)))<10  and N0*np.e**(-Parent_Dc*j)< 10:
                  break

              j+=0.01

          y=[]


          for n,m in zip(Parent_X,Daughter_X):
              y.append(n+m)

          plt.plot(t,Parent_X , label = str("Parent_"+Parent_List[graph_num-1]) )
          plt.plot(t , Daughter_X , label = CDf['Daughter_1'] )
          plt.plot(t,y)
          plt.legend()
          plt.show()
          
      except:
          print("Error")





#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~END OF FUNCTIONS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
