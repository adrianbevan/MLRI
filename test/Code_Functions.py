import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
import os
import sys
import platform
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

def Get_OS(Operating_system = platform.system()):

    file_paths = {      'CSV_filename'          :'\More_Isotopes_csv.csv',
                        'Example_Seconds'       :'\Example_Models\Example_Model_Seconds_10k_NoSkips',
                        'Example_Minutes'       :'\Example_Models\Example_Model_Minutes_10k_NoSkips',
                        'Example_Hours'         :'\Example_Models\Example_Model_Hours_10k_NoSkips',
                        'Example_Days'          :'\Example_Models\Example_Model_Days_10k_NoSkips',
                        'Isotope_list'          :'\Isotope_Half_Lifes.csv',
                        'Example_Seconds_DD'    : '\Example_Models\Example_Model_Seconds_1Yr_500Skips_Daughter_Data'                        
                    }

   
    path = os.path.abspath(os.path.dirname(sys.argv[0])) 

    if Operating_system == 'Windows':
        path = path.replace('/', '\\')


    elif Operating_system == 'Linux':
        path = path.replace('\\', '/')
        for string in  file_paths:
            file_paths[string] = file_paths[string].replace('\\', '/' )

    elif Operating_system == 'Darwin':
        Operating_system = 'Mac'
        path = path.replace('\\', '/')
        for string in  file_paths:
            file_paths[string] = file_paths[string].replace('\\', '/' )

    return path , file_paths , Operating_system


    

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


def Read_File( filename , Unit_Of_Time = 'Seconds' ):
  
  DF = pd.read_csv(filepath_or_buffer = filename)

  Isotope_List = DF['Isotope']


  DF = Set_Unit_Of_Time( DF , Unit_Of_Time = Unit_Of_Time) 


  return DF , Isotope_List 


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#This Function adds gaussian noise to a set of data when creatiing the digital twin.

#'Data' is the input data that the noise is going to be apply to , 'mu' is the mean , and 'std' is standard deviation.  


def gaussian_noise( Data , mu = 0.0 , std = 0.01 ):

  if Data < 1:
    return float(Data)

  noise = np.random.normal( mu , std, size = 1)

  Noisy_Data = Data + noise

  if Noisy_Data <0:

    return gaussian_noise( Data , mu = 0.0 , std = std)

  if Noisy_Data == 0:
    Noisy_Data = 1*(10**-21)

  return float(Noisy_Data)



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


#This function adds guassian noise to the data which is used to create a digital twin.



def Bateman_equation( X0 , Decay_constants , t = None ):
  
  X = []
  
  while True:

    number_of_decays=len(Decay_constants)
  
    if number_of_decays == 1:

        X.append( X0 * np.e**(-Decay_constants[0]*t) )

        break

    else:
      decay_product = np.prod([ i for i in Decay_constants[:number_of_decays-1] ])
      summarization = []
      for i in Decay_constants:

          summarization.append( (np.e**(-i*t)) / np.prod(  [ j - i  for j in Decay_constants if j!=i ] )  ) 

      X.append( X0 * decay_product * sum(summarization) )

      Decay_constants = Decay_constants[:len(Decay_constants)-1]
    
  return list(reversed(X))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def get_decay_chain( dataframe = None , Unit_Of_Time = None ,chain_num = 0 ):

  Decay_Identification = { isotope : decay_time for isotope , decay_time in zip( dataframe['Isotope'] , dataframe[ 'Half_Life({:})'.format(Unit_Of_Time)] ) }
  #might be an issue here (check later)#

  name_links = { isotope : daughter for isotope, daughter  in zip(dataframe['Isotope'], dataframe['Daughter']) }
  Decay_type = { isotope : daughter for isotope, daughter  in zip(dataframe['Isotope'], dataframe['Type_of_Decay']) }

  decay_name_chain = { isotope : [isotope] for isotope in  dataframe['Isotope']}

  type_of_decay_chain = { isotope : [] for isotope  in dataframe['Isotope']  }
  Half_Life_chain = { isotope : [] for isotope in  dataframe['Isotope']}

  for i in range(chain_num):

    for isotope in decay_name_chain:

      daughter_isotope = decay_name_chain[isotope][-1]

      if daughter_isotope in dataframe['Isotope'].values:

        decay_name_chain[isotope].append( name_links [ daughter_isotope ]  )

  for key in decay_name_chain:

    isotope_chain_list = decay_name_chain[key]

    for isotope in isotope_chain_list:

      if isotope in dataframe['Isotope'].values:

        Half_Life_chain[key].append(Decay_Identification[isotope])

        # if Decay_type[isotope] not in type_of_decay_chain[key]:
        type_of_decay_chain[key].append(Decay_type[isotope])

      else:
        pass
  
  Decay_const= [list(map(lambda x : (np.log(2)/x) , isotope)) for isotope in list(Half_Life_chain.values()) ]

  unique_decay_types = dataframe.Type_of_Decay.unique()
  unique_decay_types = sorted(unique_decay_types)
  #22 is max chain
  return Decay_const , type_of_decay_chain , unique_decay_types


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#



def Set_Unit_Of_Time( DF , Unit_Of_Time = None , Csv_Col='Half_Life'):

  if Csv_Col == 'Half_Life':
      Original_Time = list(DF.columns)[1]


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


def Get_Time_List(List_Type = 'Long'):
    
  if List_Type == 'Long':
    time_list = [0,315576000]

    for i in range(11):
      append_cnt=0
      while append_cnt <=1000:
        t = np.random.rand()*(10**i)
        
        while t<1 :
          t = t*10
        
        if t<315576000:
          [time_list.append( int(t) ) for j in range(1) if int(t) not in time_list ]
          append_cnt+=1
          

         
                 
    time_list = sorted(time_list)   
     

  elif List_Type == 'Random':
    time_list = [0,315576000]

    [ time_list.append( np.random.randint(0,315576000) ) for i in range( 1000 )]

  elif List_Type == 'Specific':

    while True:
      refresh()
      try:
        Min_time = float(input("Enter Minimum  Value : "))
        Max_time = float(input("Enter Maximum Value : "))
        Steps = float(input("Enter Steps Value : "))
        if Steps == float(0) : 
          print('Error: Steps cannot be equal to 0')
          time.sleep(3)
          continue
        break
      except:
        pass

    time_list = np.arange( Min_time , Max_time , Steps )


  return time_list


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


def Create_Data_Set( DF , std =0.01 , Num_Of_Replicates = 0 , Unit_Of_Time = 'Seconds' , decay_chain=0 , List_Type = 'Long' , original_Data = None , recurring_DF = None ) :

    if str(type(recurring_DF)) != "<class 'pandas.core.frame.DataFrame'>" :
      original_Data = DF
    else:
      DF = original_Data
      DF['Half_Life({:})'.format(Unit_Of_Time)] = [gaussian_noise( Data= half_life , mu = 0.0 , std = 0.01 )  for half_life in list(DF['Half_Life({:})'.format(Unit_Of_Time)]) ]

    print("Creating Data Set Please Wait... ({:} Replications Left)".format(Num_Of_Replicates))
    

    time_list = Get_Time_List(List_Type = List_Type)
    Decay_const , type_of_decay_chain ,unique_decay_types  = get_decay_chain( dataframe = DF , Unit_Of_Time = Unit_Of_Time , chain_num = decay_chain )

    N0=1
    DF ={ 'N'           : [],
          't'           : [],
          'Decay_Type'  : [],
          'Isotope'     : [] }


    Decay_const ,type_of_decay_chain = Decay_const[680:] ,list(type_of_decay_chain.values())[680:]
    All_decay_types=[]
    [ All_decay_types.append( i ) for i in type_of_decay_chain  ]


    Numerical_Decay_Types = { Decay : Num for Num , Decay in enumerate(unique_decay_types , start = 1) }



    Progression( 0 , len(Decay_const) )                                # range should be going from one when i appple the stable isotope identifications.
    for isotope_decay_chain , index , decay_types in zip( Decay_const, range(len(Decay_const)) , All_decay_types ):

      

      for t in time_list:
        
        Specific_Decay_Sum = { i : [] for i in np.unique(decay_types) }
        Ns = Bateman_equation( X0 = N0 , Decay_constants = isotope_decay_chain , t=t )
        
        if sum(Ns)!=0:

          for  i , N in enumerate(Ns) :
  
            Specific_Decay_Sum[ decay_types[i] ].append( N )
  
          for Decay_Sum in Specific_Decay_Sum:
            if N !=0:
              N = sum( Specific_Decay_Sum[Decay_Sum] )
  
              DF['N'].append(N)
              DF['t'].append(t)
              DF['Decay_Type'].append(Numerical_Decay_Types[ Decay_Sum ])
              DF['Isotope'].append(index)
              
        #elif N ==0:
        #  DF['N'].append(sum(Ns))
        #  DF['t'].append(t)
        #  DF['Decay_Type'].append( 0 )
        #  DF['Isotope'].append(index)
                

      Progression(index+1 , len(Decay_const) )



    DF = pd.DataFrame(DF)

    if Num_Of_Replicates == 0:
      if str(type(recurring_DF)) != "<class 'pandas.core.frame.DataFrame'>" :
        return DF
      else:
        recurring_DF = pd.concat([ recurring_DF , DF])
        DF = None
        recurring_DF.reset_index(drop=True, inplace=True)
        return recurring_DF


    
    else:

      if str(type(recurring_DF)) != "<class 'pandas.core.frame.DataFrame'>" :
        recurring_DF = DF
        DF = None
        refresh()
        return Create_Data_Set( DF = original_Data  , std = std , Num_Of_Replicates = Num_Of_Replicates-1 , Unit_Of_Time = 'Seconds' , decay_chain=decay_chain , List_Type = 'Long' , original_Data = original_Data , recurring_DF = recurring_DF )

      else:
        recurring_DF = pd.concat([ recurring_DF , DF])
        DF = None
        refresh()
        return Create_Data_Set( DF = original_Data  , std = std , Num_Of_Replicates = Num_Of_Replicates-1 , Unit_Of_Time = 'Seconds' , decay_chain=decay_chain , List_Type = 'Long' , original_Data = original_Data , recurring_DF = recurring_DF )






#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#



def Plot_Data_Frame( filename , Unit_Of_Time = 'Seconds' , decay_chain=0 , Specific_Radioisotope = False):

  if Specific_Radioisotope == False:
    print("Plotting Data Please Wait...")

    DF , Isotope_List  = Read_File( filename = filename , Unit_Of_Time = Unit_Of_Time )

    time_list = Get_Time_List( List_Type = 'Long' )

    Decay_const , type_of_decay_chain , unique_decay_types = get_decay_chain( dataframe = DF , Unit_Of_Time = Unit_Of_Time , chain_num = decay_chain )
    Decay_const = Decay_const[680:]
    N0=1
    Progression(0 , len(Decay_const) )
    for isotope , index in zip(Decay_const,range(len(Decay_const))):
      Ns=[]
      for t in time_list:

        N = Bateman_equation( X0 = N0 , Decay_constants= isotope , t=t )

        Ns.append(sum(N))
        
      Progression(index+1 , len(Decay_const) )
      plt.ylim(0, 1.2)
      plt.plot(time_list,Ns)

    plt.show()

  else:
    DF , Isotope_List  = Read_File( filename = filename , Unit_Of_Time = Unit_Of_Time )
    Shopping_List = Isotope_Shopping_List(Isotope_List = Isotope_List, Shopping_List = [])
    
    if Shopping_List == None:
      return
    
    for index,isotope in enumerate(Shopping_List ):

      print(f"Plotting Data Please Wait...{Isotope_List[isotope]}")

      time_list = Get_Time_List( List_Type = 'Long' )

      Decay_const , type_of_decay_chain , unique_decay_types = get_decay_chain( dataframe = DF , Unit_Of_Time = Unit_Of_Time , chain_num = decay_chain )
      Decay_const = Decay_const[isotope-3:isotope-2][0]


      plot_dict = { i : [] for i in range(len(Decay_const)) }

      cnt=0
      x_axis =[]
      for t in time_list:

        Ns = Bateman_equation( X0 = 1 , Decay_constants = Decay_const , t=t )


        [ plot_dict[i].append(Dc) for i , Dc in enumerate(Ns) ]
        x_axis.append(t)

        if sum(Ns) == 0: break
      
      for i in plot_dict:

        plt.ylim(0, 1.2)
        plt.plot(x_axis,plot_dict[i])

      plt.xlabel(f"time in {Unit_Of_Time}")
      plt.ylabel(f"N")
      plt.title(str(Isotope_List[isotope]))
      plt.show()






#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


def training_V2 ( Training_df  ,Isotope_List , New_Model=True , model = None , Training_Logs = False , Operating_system =None):

    if New_Model== True and Training_Logs == False: 

        train_labels = Training_df.pop('Isotope')

        Training_df ,train_labels = shuffle(Training_df ,train_labels )

        scaler = MinMaxScaler()

        Training_df[['t','Decay_Type']] = scaler.fit_transform(Training_df[['t','Decay_Type']])


        model = Sequential([Dense(512 , input_shape=((Training_df.shape)[1],) ,activation='sigmoid'),
                            Dense(512 , activation='sigmoid'),
                            #Dense(38 , activation='sigmoid'),
                            #Dense(38 , activation='sigmoid'),
                            Dense(len(train_labels.unique()),activation='softmax')])


        model.compile(optimizer=Adam(learning_rate=0.01), 
                    loss='sparse_categorical_crossentropy', 
                    metrics=['accuracy',])

        print(model.summary())

        history  = model.fit(   Training_df, train_labels , 

                                validation_split=0.1 , epochs=50 ,

                                shuffle=True , verbose=2) 


        return model , history 


    elif New_Model == True and Training_Logs == True:

        Log_Date = time.ctime(time.time()).replace(':','_').replace(' ','_')

        while True:
            Num_Of_Epochs = input("Enter Number of Epochs : ")
            if Num_Of_Epochs.isdigit() == True:
                Num_Of_Epochs = int(Num_Of_Epochs)
                break

        train_labels = Training_df.pop('Isotope')

        Training_df ,train_labels = shuffle(Training_df ,train_labels )

        scaler = MinMaxScaler()

        Training_df[['t','Decay_Type']] = scaler.fit_transform(Training_df[['t','Decay_Type']])


        model = Sequential([Dense(256, input_shape=((Training_df.shape)[1],) ,activation='sigmoid'),
                            Dense(256, activation='sigmoid'),
                            Dense(len(train_labels.unique()),activation='softmax')])


        model.compile(optimizer=Adam(learning_rate=0.01), 
                    loss='sparse_categorical_crossentropy', 
                    metrics=['accuracy',])

        print(model.summary())
        history  = model.fit( Training_df, train_labels , validation_split=0.1 , epochs=1 , shuffle=True , verbose=2)


            
        if Operating_system == 'Windows':
            model.save("Model_Log_{:}\{:}_Epoch_{:}".format( Log_Date,Log_Date, 1 ))
            
            with open("Model_Log_{:}\Model_Logs_{:}.txt".format(Log_Date,Log_Date),'a') as file:
              model.summary(print_fn=lambda x: file.write(x + '\n'))
              file.write("\n{:} - Epoch_{:} ".format( history.history , 1 ))
              
        else:
            model.save("Model_Log_{:}/{:}_Epoch_{:}".format( Log_Date,Log_Date, 1 ))
            
            with open("Model_Log_{:}/Model_Logs_{:}.txt".format(Log_Date,Log_Date),'a') as file:
              model.summary(print_fn=lambda x: file.write(x + '\n'))
              file.write("\n{:} - Epoch_{:} ".format( history.history , 1 ))


        for  i in  range( 2 , 1 + Num_Of_Epochs  ) : 


            Training_df ,train_labels = shuffle(Training_df ,train_labels )
            
            history  = model.fit(   Training_df, train_labels , validation_split=0.1 , epochs=1 , shuffle=True , verbose=2) 


            
            if Operating_system == 'Windows':
                model.save("Model_Log_{:}\{:}_Epoch_{:}".format( Log_Date,Log_Date, i ))
                with open("Model_Log_{:}\Model_Logs_{:}.txt".format(Log_Date,Log_Date),'a') as file:
                  file.write("\n{:} - Epoch_{:} ".format( history.history , i ))
            else:
                model.save("Model_Log_{:}/{:}_Epoch_{:}".format( Log_Date,Log_Date, i ))
                with open("Model_Log_{:}/Model_Logs_{:}.txt".format(Log_Date,Log_Date),'a') as file:
                  file.write("\n{:} - Epoch_{:} ".format( history.history , i ))

            




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

  
  df_test_eval = Testing_df.pop('Isotope')
  Testing_df[['t','Decay_Type']] = scaler.fit_transform(Testing_df[['t','Decay_Type']])
  scaled_df_test,df_test_eval = shuffle(Testing_df,df_test_eval)
  print("Isotope popped")




  eval_result = model.predict(x=scaled_df_test)
    
  print("Evalled result")
  

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


  Original_Type_Of_Decay= DF.pop('Type_of_Decay_1')

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
      N = Bateman_equation( N0 ,decay_constant , t = t )
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
def Progression(possision , termination):
  progress = 100 * (possision / float(termination))
  bar = str('~'*(100-(100-int(progress)))+'→' +' ' * (100-int(progress)))
  # print("\rPlease Wait...",end='\n')
  print( f"\r¦{bar}¦ {progress:.0f}%",end='')

  




#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def Load_DataFrame():
  DF = pd.read_pickle(input("Enter file name or path : "))

  return DF


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#



def Specific_model():
  loaded_model = load_model(input("Enter file name or path : "))
  loaded_model.summary()

  return loaded_model

def Example_model(filename):
  loaded_model = load_model(filename)
  loaded_model.summary()

  return loaded_model



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~END OF FUNCTIONS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
