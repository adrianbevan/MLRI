import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as mplcm
import numpy as np
import pandas as pd
import seaborn as sn
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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# "refrsh()" Is used to clear the terminal when using the user interface # 

def refresh():
  clear_output()
  os.system('cls||clear')
  time.sleep(0.5)   



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# "Get_OS" gets the system operating system that is executing the code and returns "path" , "file_paths" , "Operating_system".
# "path" is the location of the directory that the scripts are stored in.
# "file_paths" are the locations of specific files relative the the "path".



def Get_OS(Operating_system = platform.system()):

    file_paths = {      'CSV_filename'          :'\Isotope_Half_Lifes_CSV.csv',
                        'Example_Seconds'       :'\Example_Model_Seconds_0_Decay_Chain',
                        'Isotope_List'          :'\Text_Files\Isotope_List.txt'               
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

# "Read_File()"  takes two arguments "filename" and "Unit_Of_Time".
# "filename" is the file locative of the CSV file containing all of the isotope half-lifes , daughters and types of decay.
# "Unit_Of_Time" is the time that the user wants the data to be in "Seconds", "Minutes" , "Hours" and "Days".
# 
# The funtion returns two variable "DF" and "Isotope_List".
# "DF" is a pandas DataFrame of the CSV file.
# "Isotope_List" is a list of all the names of parent radioactive nuclei from the CSV file.


def Read_File( filename , Unit_Of_Time = 'Seconds' ):
  
  DF = pd.read_csv(filepath_or_buffer = filename)

  Isotope_List = DF['Isotope']


  DF = Set_Unit_Of_Time( DF , Unit_Of_Time = Unit_Of_Time) 


  return DF , Isotope_List 


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#This Function adds gaussian noise to a set of data when creatiing the digital twin.

#'Data' is the input data that the noise is going to be apply to  and 'std' is standard deviation in percent.  


def Add_Noise( Data  , std = 1 ):

  Data = np.array( float(Data) ) 

  Noisy_Data = np.random.normal( Data , std , 1 )[0]

  return Noisy_Data


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# "Bateman_equation()" is the function that calculates the abundance/activity of a radioactive isotope at a given time  and takes the arguments "X0" , "Decay_constants" , "t".
# "X0" is the initial count or activity of the parent nuclei.
# "Decay_constants" is a list of all decay constants of radioactive nuclei in the decay chain for a given parent.
# "t" is the time. 
#
# The function returns a list of all of the counts/activties of each radioactive nuclie in the decay chain in order from parent to last daughter in decay chain. 


def Bateman_equation( X0 , Decay_constants , t = None ):
  
  X = []
  X0 = X0*(10**100)
  e  = np.e
  while True:

    number_of_decays=len(Decay_constants)
  
    if number_of_decays == 1:

        X.append( X0 * e**(-Decay_constants[0]*t) )

        break

    else:
      decay_product = np.prod([ i for i in Decay_constants[:number_of_decays-1] ])
      summarization = []
      
      product_optimizer =[]
      
      for i in Decay_constants:

          product_optimizer.append(   [ j - i  for j in Decay_constants if j!=i ]   ) 
          

      product_optimizer = np.prod( product_optimizer , axis = 1 ) 
       
      for i,prdct in zip(Decay_constants,product_optimizer) :
      
        summarization.append( (e**(-i*t)) / prdct )
      
        

      X.append( X0 * decay_product * sum(summarization) )

      Decay_constants = Decay_constants[:len(Decay_constants)-1]
  
  X = list(map(lambda x : x/(10**100) , X) )
     
  return list(reversed(X))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# "get_decay_chain()" finds all the the radioactive children in a decay chain for a radioactive parent, and returns "Decay_const" , "type_of_decay_chain" , "unique_decay_types" , "decay_name_chain".
# The argument "dataframe" is the pandas DataFrame returned from the "Read_File()" funtion. 
# "Unit_Of_Time" is the unit of time decided by the user "Seconds", "Minutes" , "Hours" and "Days".
# "chain_num" is the length of how long the user want the decay chain to be, if "0" there will only be using data from the parent ... if "2" it will be using data from the parent , daughter and granddaughter.
#
# "Decay_const" is a list of list of all the decay chain constants for every single radioactive isotope in the Dataframe (from CSV file ).
# "type_of_decay_chain" is a list of list of all the types of decay (example : ["alpha" , "beta +" , "alpha" ... "beta -"]) for every single radioactive isotope in the Dataframe (from CSV file )
# "decay_name_chain" is a list of list of all the names of isotopes in the decay chain for every single radioactive isotope in the Dataframe (from CSV file )

def get_decay_chain( dataframe = None , Unit_Of_Time = None ,chain_num = 0 ):

  Decay_Identification = { isotope : decay_time for isotope , decay_time in zip( dataframe['Isotope'] , dataframe[ 'Half_Life({:})'.format(Unit_Of_Time)] ) }

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

        type_of_decay_chain[key].append(Decay_type[isotope])

      else:
        pass
  
  Decay_const= [list(map(lambda x : (np.log(2)/x) , isotope)) for isotope in list(Half_Life_chain.values()) ]

  unique_decay_types = dataframe.Type_of_Decay.unique()
  unique_decay_types = sorted(unique_decay_types)

  return Decay_const , type_of_decay_chain , unique_decay_types , decay_name_chain


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# "Set_Unit_Of_Time()" allows the user to switch unit of time that program is using , as the CSV file hale-lifes are in seconds and this may want to be changed to "Minutes" , "Hours" or "Days". 
# The argument "DF" is the pandas DataFrame returned from the "Read_File()" funtion. 
# "Unit_Of_Time" is the unit of time decided by the user "Seconds", "Minutes" , "Hours" and "Days".
# "Csv_Col" is part of the lable of the column that you can make changes to.
# 
# "DF" that is returned is the is dataframe that is inputed as a argument but the values and the column name for the Hale-Life have been changed to the users preference.


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

# "Get_Time_List()" if a funtion that returns a list of random integers that represent times.
# It takes one argument "List_Type" and this can be equal to ('Long') , ('Random') or ('Specific')
# When "List_Type" is "Long"  it will return a list with about 50,000 random values between 0-315576000 which is essentially 0 and 10 years in seconds.
# When "List_Type" is "Random"  it will return a list with about 5,000 random values.
# When "List_Type" is "Specific"  it will alllow the user to input the minimun and maximum values aswell as the number of steps.



def Get_Time_List(List_Type = 'Long'):
  print("Simulating Time...")
    
  if List_Type == 'Long':
    time_list = [0,315576000]
    Progression( 0 , 11)
    
    for i in range(11):
      random_list = np.random.rand(1,1*10**4)[0]
      for t  in random_list :
        t = np.random.rand()*(10**i)
        
        while t<1 :
          t = t*10
        
        if t<315576000:
          [time_list.append( int(t) ) for j in range(1) if int(t) not in time_list ]
          
          
      Progression( i , 11)
    
    time_list = sorted(time_list)
    refresh()   
 
     

  elif List_Type == 'Random':
    time_list = [0,315576000]

    [ time_list.append( np.random.randint(0,315576000) ) for i in range( 5000 )]

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

# "Create_Data_Set()" takes in the the original pandas DataFrame and other arguments and outputs a pandas dataframe based on the users configuration that can be used to train and evaluate the Tensorflow models.
# "DF" is the pandas DataFrame returned from the "Read_File()" funtion.
# "std" is the standard deviation (in percent) on the half-lifes if guassion noise is applied to the dataframe when replicating.
# "Num_Of_Replicates" is the number of times the "Create_Data_Set()" funtion will be recursive, each time guassion noise will be added to the Half-Life.
# "Unit_Of_Time" is the unit of time decided by the user "Seconds", "Minutes" , "Hours" and "Days".
# "decay_chain" is the length of how long the user want the decay chain to be, this will be used in the "get_decay_chain()" function.
# "List_Type" decides the type of list returned from the "Get_Time_List()" funtion.
# "original_Data" is a placement variable that holds the data from "DF" when the function is being recursive.
# "recurring_DF" is a placement variable that holds the returned dataframe thats is used to get the next returned dataframe appended to it.
# "Decay_Seperation" can either be "True" or "False". When "True" the data set will split all the types of decay apart for training , when "False" the data set will summarize all the different counts/activities and append a random tpye of decay at each instance in time.
# "Shopping_List" is the list of specific isotopes the user is looking for.
# 
# If "Num_Of_Replicates" = 0 then the function will return the first data set created.
# If "Num_Of_Replicates" > 0 then the function will return the recurring dataframe with all the data sets appended. 

def Create_Data_Set( DF , std = 1  , Num_Of_Replicates = 0 , Unit_Of_Time = 'Seconds' , decay_chain=0 , List_Type = 'Long' , original_Data = None , recurring_DF = None , Decay_Seperation = False ,Shopping_List = [] ) :

    if str(type(recurring_DF)) != "<class 'pandas.core.frame.DataFrame'>" :
      original_Data = DF
    else:
      DF = original_Data
      DF['Half_Life({:})'.format(Unit_Of_Time)] = [Add_Noise( Data= half_life , std = 1 )  for half_life in list(DF['Half_Life({:})'.format(Unit_Of_Time)]) ]

    time_list = Get_Time_List(List_Type = List_Type)
    
    print("Creating Data Set Please Wait... ({:} Replications Left)".format(Num_Of_Replicates))
    

    
    Decay_const , type_of_decay_chain , unique_decay_types , decay_name_chain  = get_decay_chain( dataframe = DF , Unit_Of_Time = Unit_Of_Time , chain_num = decay_chain )

    N0=1
    DF ={ 'N'           : [],
          't'           : [],
          'Decay_Type'  : [],
          'Isotope'     : [] }

    if Shopping_List == [] :
      Decay_const ,type_of_decay_chain = Decay_const[679:848] ,list(type_of_decay_chain.values())[679:848]
    else :
      Decay_const_holder         = []
      type_of_decay_chain_holder = []
      for  i in  Shopping_List:
        Decay_const_holder.append(Decay_const[i])
        type_of_decay_chain_holder.append( list(type_of_decay_chain.values())[i])

      
      Decay_const ,type_of_decay_chain = Decay_const_holder , type_of_decay_chain_holder

    All_decay_types=[]
    [ All_decay_types.append( i ) for i in type_of_decay_chain  ]


    Numerical_Decay_Types = { Decay : Num for Num , Decay in enumerate(unique_decay_types , start = 1) }



    Progression( 0 , len(Decay_const) )                            
    for isotope_decay_chain , index , decay_types in zip( Decay_const, range(len(Decay_const)) , All_decay_types ):

      for t in time_list:
        
        Specific_Decay_Sum = { i : [] for i in decay_types }
        Ns = Bateman_equation( X0 = N0 , Decay_constants = isotope_decay_chain , t=t )
        
        if sum(Ns)!=0:

          for  i , N in enumerate(Ns) :
    
            Specific_Decay_Sum[ decay_types[i] ].append( N )

          if Decay_Seperation == False:
            total_abundance = 0
            random_type_of_decay = np.random.choice( list(Specific_Decay_Sum.keys()))

            for Decay_Sum in Specific_Decay_Sum:

              total_abundance += sum( Specific_Decay_Sum[Decay_Sum] )

            DF['N'].append(total_abundance)                          
            DF['t'].append(t)
            DF['Decay_Type'].append(Numerical_Decay_Types[ random_type_of_decay ])
            DF['Isotope'].append(index)


          elif Decay_Seperation == True:

            for Decay_Sum in Specific_Decay_Sum:
              total_abundance = sum( Specific_Decay_Sum[Decay_Sum] )
              
              if total_abundance !=0:                
        
                DF['N'].append(total_abundance)                          
                DF['t'].append(t)
                DF['Decay_Type'].append(Numerical_Decay_Types[ Decay_Sum ])
                DF['Isotope'].append(index)

        
        

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
        return Create_Data_Set( DF = original_Data  , std = std , Num_Of_Replicates = Num_Of_Replicates-1 , Unit_Of_Time = Unit_Of_Time , decay_chain=decay_chain , List_Type = 'Long' , original_Data = original_Data , recurring_DF = recurring_DF , Decay_Seperation = Decay_Seperation ,Shopping_List = Shopping_List )

      else:
        recurring_DF = pd.concat([ recurring_DF , DF])
        DF = None
        refresh()
        return Create_Data_Set( DF = original_Data  , std = std , Num_Of_Replicates = Num_Of_Replicates-1 , Unit_Of_Time = Unit_Of_Time , decay_chain=decay_chain , List_Type = 'Long' , original_Data = original_Data , recurring_DF = recurring_DF , Decay_Seperation = Decay_Seperation ,Shopping_List = Shopping_List )






#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# "Plot_Data_Frame()" is a function that allows users to plot the decay curves for one or multiple radioactive isotopes.
# "filename" is the file locative of the CSV file containing all of the isotope half-lifes , daughters and types of decay.
# "Unit_Of_Time" is the time that the user wants the data to be in "Seconds", "Minutes" , "Hours" and "Days".
# "decay_chain" is the length of how long the user want the decay chain to be, this will be used in the "get_decay_chain()" function.
# "Specific_Radioisotope" can be either "True" or "False". 
# When True it will promt the user to plot which radioactive isotope they want to plot the decay curves to.
# if "False" it will plot the total count/activity curves for multiple radioactive isotopes.

def Plot_Data_Frame( filename , Unit_Of_Time = 'Seconds' , decay_chain=0 , Specific_Radioisotope = False , path = "" ,  Shopping_List = [] ):

  DF , Isotope_List  = Read_File( filename = filename , Unit_Of_Time = Unit_Of_Time )

  if Specific_Radioisotope == False:
    time_list = Get_Time_List( List_Type = 'Long' )
    print("Plotting Data Please Wait...")

    Decay_const , type_of_decay_chain , unique_decay_types , decay_name_chain = get_decay_chain( dataframe = DF , Unit_Of_Time = Unit_Of_Time , chain_num = decay_chain )

    if Shopping_List == [] :
      Decay_const ,type_of_decay_chain = Decay_const[679:848] ,list(type_of_decay_chain.values())[679:848]
    else :
      Decay_const_holder         = []
      type_of_decay_chain_holder = []
      decay_name_chain_holder = []
      for  i in  Shopping_List:
        Decay_const_holder.append(Decay_const[i])
        type_of_decay_chain_holder.append( list(type_of_decay_chain.values())[i])
        decay_name_chain_holder.append( list(decay_name_chain.values())[i][0])


      Decay_const ,type_of_decay_chain ,  decay_name_chain = Decay_const_holder , type_of_decay_chain_holder , decay_name_chain_holder

    N0=1
    Progression(0 , len(Decay_const) )

    Number_Of_Colours = len( list(Decay_const ) )
    color_map = plt.get_cmap('gist_rainbow')
    cNorm  = colors.Normalize(vmin=0, vmax=Number_Of_Colours-1)
    scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=color_map)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_prop_cycle( color = [color_map(1.*i/Number_Of_Colours) for i in range(Number_Of_Colours)] )
    plt.rc('font', size=30) 
    for isotope , index in zip(Decay_const,range(len(Decay_const))):
      Ns=[]
      
      for t in time_list:

        N = Bateman_equation( X0 = N0 , Decay_constants= isotope , t=t )

        Ns.append(sum(N))
        
      Progression(index+1 , len(Decay_const) )
      plt.ylim(0, 1.2)
      if Shopping_List == []:
        plt.plot(time_list , Ns )
      else:
        plt.plot(time_list , Ns , label = decay_name_chain[index])
      


    if Shopping_List != []:
      plt.rc('legend',fontsize=20)
      plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
      plt.title("Radioactive Decays of Isotopes In Example Lab Setting")
    plt.xlabel("Time in {:}".format(Unit_Of_Time))
    plt.ylabel("Total Activity")
    plt.show()

  else:
    
    Shopping_List = Isotope_Shopping_List(Isotope_List = Isotope_List, Shopping_List = [] , path = path)
    
    if Shopping_List == None:
      return
    refresh()
    for index , isotope in enumerate( Shopping_List ):

      print("Plotting Data Please Wait...{}".format(Isotope_List[isotope]))

      time_list = Get_Time_List( List_Type = 'Long' )

      Decay_const , type_of_decay_chain , unique_decay_types , decay_name_chain = get_decay_chain( dataframe = DF , Unit_Of_Time = Unit_Of_Time , chain_num = decay_chain )
      Decay_const = Decay_const[isotope]
      type_of_decay_chain = list(type_of_decay_chain.values())[isotope]
      decay_name_chain = list(decay_name_chain.values())[isotope]

      plot_dict = { i : [] for i in range(len(Decay_const)) }

      cnt=0
      x_axis =[]

      Number_Of_Colours = len( list(plot_dict.keys() ) )
      color_map = plt.get_cmap('gist_rainbow')
      cNorm  = colors.Normalize(vmin=0, vmax=Number_Of_Colours-1)
      scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=color_map)
      fig = plt.figure()
      ax = fig.add_subplot(111)
      ax.set_prop_cycle( color = [color_map(1.*i/Number_Of_Colours) for i in range(Number_Of_Colours)] )

      for t in time_list:

        Ns = Bateman_equation( X0 = 1 , Decay_constants = Decay_const , t=t )


        [ plot_dict[i].append(Dc) for i , Dc in enumerate(Ns) ]
        x_axis.append(t)

        if sum(Ns) == 0: break
      
      for i , t_o_d ,d_n_c in zip( plot_dict , type_of_decay_chain , decay_name_chain):

        plt.ylim(0, 1.2)
        plt.plot( x_axis , plot_dict[i] , label = "{:}  :  {:}".format( str(d_n_c), str(t_o_d)))

      plt.rc('font', size=30)
      plt.rc('legend',fontsize=20)
      plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
      plt.xlabel(f"time in {Unit_Of_Time}")
      plt.ylabel(f"N")
      plt.title(str(Isotope_List[isotope]))
      plt.show()






#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# "Training_V2" , is responsible for training the algorithm .
# "Training_df" is the data set returned from "Create_Data_Set()", all the values will be scaled between 0 and 1 before training.
# "Config_Receipt" is a dictionary containing all on the configurations by the user when "Training Logs" are set in the the User Interface.
# "New_Model" can either be True or False. When True the program will create and train a new model, when false the program will continue to train the existing model.
# "model" is a placeholder for and existing model if one is already trained.
# If "Training_Logs" is true the program will create text file that will store the model accuracy on each epoch trained.
# "Operating_system" is the operating system returned from "Get_OS()".

def Training_V2 ( Training_df  , Config_Receipt = {} , New_Model=True , model = None , Training_Logs = False , Operating_system =None):

    if New_Model== True and Training_Logs == False: 

        while True:
            Num_Of_Epochs = input("Enter Number of Epochs : ")
            if Num_Of_Epochs.isdigit() == True:
                Num_Of_Epochs = int(Num_Of_Epochs)
                break

        train_labels = Training_df.pop('Isotope')

        Training_df ,train_labels = shuffle(Training_df ,train_labels )

        scaler = MinMaxScaler()

        Training_df[['t','Decay_Type']] = scaler.fit_transform(Training_df[['t','Decay_Type']])



        model = Sequential([Dense(16 , input_shape=((Training_df.shape)[1],) ,activation='sigmoid'),
                            Dense(32 , activation='sigmoid'),
                            Dense(len(train_labels.unique()),activation='softmax')])


        model.compile(optimizer=Adam(learning_rate=0.01), 
                    loss='sparse_categorical_crossentropy', 
                    metrics=['accuracy',])
        
        print("\n{:}\n\n".format(Config_Receipt))
        print(model.summary())

        history  = model.fit(   Training_df, train_labels , 

                                validation_split=0.1 , epochs=Num_Of_Epochs ,

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
        


        model = Sequential([Dense(16, input_shape=((Training_df.shape)[1],) ,activation='sigmoid'),
                            Dense(32, activation='sigmoid'),
                            Dense(len(train_labels.unique()),activation='softmax')])


        model.compile(optimizer=Adam(learning_rate=0.01), 
                    loss='sparse_categorical_crossentropy', 
                    metrics=['accuracy',])
                    
        print("\n{:}\n\n".format(Config_Receipt))
        print(model.summary())
        
        history  = model.fit( Training_df, train_labels , validation_split=0.1 , epochs=1 , shuffle=True , verbose=2)


            
        if Operating_system == 'Windows':
            model.save("Model_Log_{:}\{:}_Epoch_{:}".format( Log_Date,Log_Date, 1 ))
            
            with open("Model_Log_{:}\Model_Logs_{:}.txt".format(Log_Date,Log_Date),'a') as file:
              model.summary(print_fn=lambda x: file.write(x + '\n'))

              for Data_Name in Config_Receipt :
                file.write("\n{:} : {:}\n".format( Data_Name , Config_Receipt[Data_Name] ))

              file.write("\n{:} - Epoch_{:} ".format( history.history , 1 ))
              
        else:
            model.save("Model_Log_{:}/{:}_Epoch_{:}".format( Log_Date,Log_Date, 1 ))
            
            with open("Model_Log_{:}/Model_Logs_{:}.txt".format(Log_Date,Log_Date),'a') as file:
              model.summary(print_fn=lambda x: file.write(x + '\n'))

              for Data_Name in Config_Receipt :
                file.write("\n{:} : {:}".format( Data_Name , Config_Receipt[Data_Name] ))
                
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

# "Isotope_Shopping_List" allows selection of specific isotopes from the csv file to use throughout the program.
# "Isotope_List" is the list of all isotopes from the CSV file that is returned from "Read_File()".
# "Shopping_List" is the list of specific isotopes the user is looking for.
# "Shopping" is just to check whether the user is still choosing more isotopes, if "False" the funtion "Isotope_Shopping_List" will end.

def Isotope_Shopping_List(Isotope_List , Shopping_List=[] ,Shopping =True , path = ""):

  Isotope_Identification = { i:index for i,index in zip( Isotope_List , range(  len(Isotope_List) ) )}
  Sorted_Istope_List = np.sort(Isotope_List)


  while Shopping:

    refresh()

    for  isotope , index in zip(Sorted_Istope_List , range(  len(Sorted_Istope_List) ) )  : 

        print( '{:20s} : {:} '. format( isotope , index ) ) 

    print( '\n{:20s} : {:} '. format( "Show Isotope List" , "i" ) ) 

    

    print( '\n{:21s} : {:} '. format( "Back" , "q" ) ) 

    Item = input('\n\n{:22s} : '. format( "Option" ) )

    if Item.isdigit()==True :
        if int(Item) in range(  len(Sorted_Istope_List)  ):
          Item = Isotope_Identification [ Sorted_Istope_List [ int(Item) ]]

          if Item not in Shopping_List : Shopping_List.append(Item)

        refresh()


    elif Item == 'i':
        
        while True:

            refresh()
            Categotical_Shopping_List=[]
            for isotope in Shopping_List  :  Categotical_Shopping_List.append( Isotope_List[isotope])
            Operating_system = platform.system()
            print(Categotical_Shopping_List)
            print( '\n{:20s} : {:} '. format( "Remove Isotope" , "1" ) )
            print( '\n{:20s} : {:} '. format( "Empty Isotope List" , "2" ) )
            print( '\n{:20s} : {:} '. format( "Save Isotope List" , "3" ) )
            print( '\n{:20s} : {:} '. format( "Load Isotope List" , "4" ) )
            print( '\n\n{:20s} : {:} '. format( "Back" , "q" ) ) 

            Option = input("Option : ")

            if Option =="1":

                if Shopping_List == []:
                  pass
      
                else:
                  while True :
                    refresh()
                    print("{:30s}\n\n".format( "Pick Isotope To Remove ") )

                    for indx, isotope in enumerate( Shopping_List , start = 1):
                      print("{:20s}{:30s}\n".format( Isotope_List[isotope] , str(indx) ) )
                    
                    Option = input('\n{:20s} :  '. format( "Option" ))

                    if Option.isdigit() == True:
                      if int( Option ) in range( 1 , len(Shopping_List) + 1 ):

                        Shopping_List.remove(Shopping_List[ int(Option) - 1 ])

                        break

                    refresh()




            elif Option =="2":
                Shopping_List = [] 

            elif Option =="3":
              refresh()
              if Shopping_List != []:
                List_Name = input("Enter Name for this Isotope List : ")

                if Operating_system == "Windows":
                  with open(path+"\Text_Files\Saved_Shopping_List.txt",'a') as file:
                    file.write( "{}~{}\n".format( List_Name , Shopping_List) )
                  
                else:
                  with open(path+"/Text_Files/Saved_Shopping_List.txt",'a') as file:
                    file.write( "{}~{}\n".format( List_Name , Shopping_List) )


                  
            elif Option =="4":
              refresh()
              try:
                if Operating_system == "Windows":
                  file = open(path+"\Text_Files\Saved_Shopping_List.txt",'r').read()
                
                else:
                  file = open(path+"/Text_Files/Saved_Shopping_List.txt",'r').read()
              except:
                print("ERROR : No Found Shopping List in 'Text_Files'")
                time.sleep(3)
                break


              lines = file.splitlines()
              while True:
                for index , line in enumerate( lines , start = 1):

                  name , iso_list = line.split("~")
                  temp_Categotical_Shopping_List=[]

                  for isotope in eval(iso_list)  :  temp_Categotical_Shopping_List.append( Isotope_List[isotope])
                  print("{:20s}{:30s}   ({:})\n".format(name , str(temp_Categotical_Shopping_List) ,index) )

                print( '\n{:20s} : {:} '. format( "Back" , "q" ) ) 
              
                Option = input("Option : ")
                refresh()

                if Option.isdigit() == True:
                  if int( Option ) in range( 1 , len(lines) + 1 ):

                    str_list = lines[int(Option)-1].split('~')[1]
                    Shopping_List= eval(str_list)
                    break
                
                elif Option == 'q': 
                  refresh()
                  break
                 
            elif Option =="q":
              break    

    elif Item == 'q':
        break 

  if Shopping_List == []  : return

  else : return Shopping_List

  







#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# "Evaluate()", evaluates trained models.
# "model" is the trained model.
# If "Unknown_Isotope" is set to "False" the model will be evaluated and the model accuracy will be displayed , If "True" the model will evaluate the model no accuracy will be displayed.


def Evaluate( Testing_df , model , Unknown_Isotope = False):

  df_test_eval = None
  scaler = MinMaxScaler()

  try:
    df_test_eval = Testing_df.pop('Isotope')
  except:
    pass
  Testing_df[['t','Decay_Type']] = scaler.fit_transform(Testing_df[['t','Decay_Type']])
  try:
    scaled_df_test,df_test_eval = shuffle(Testing_df,df_test_eval)
  except:
    scaled_df_test = shuffle(Testing_df)
  print("Isotope popped")




  eval_result = model.predict(x=scaled_df_test)
    
  print("Evalled result")
  

  rounded_predictions = np.argmax(eval_result,axis=-1)

  if Unknown_Isotope == False:

      model.evaluate(x=scaled_df_test , y=df_test_eval , verbose=1 )
      

  return eval_result  , df_test_eval 

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# "Further_Evalutaion()" will display all of the model predictions at every instance of time from the time list ( from "Get_Time_List()" ).
# "eval_result" a list of numpy arrays of predictions made by the model each position in the array matches to an isotope.
# "Isotope_List" is the list of all isotopes from the CSV file that is returned from "Read_File()".
# "df_test_eval" a list of integers that matched to the indexes of the correct isotopes.
# If "Unknown_Isotope" is set to "False" the first and second predictions will be displayed along with their probabilities and if the models first guess was "Correct" or "Wrong" 
# If "Unknown_Isotope" is set to "True" only the model prediction will be displayed.
# "Radioactive_Shopping_List" is "True" if there is a shopping list and "False" if there is no shopping list.
# "Shopping_List" is the list of specific isotopes the user is looking for.
# If "Training_Logs" is true the program will create text file that contain all of the predictions instead of displaying them (faster).
# "path" is the location of the directory that the scripts are stored in.


def Further_Evalutaion ( eval_result, Isotope_List , df_test_eval=None , Unknown_Isotope = False , Radioactive_Shopping_List = False  , Shopping_List = None , Training_Logs = False , path = ""):
  Log_Date = time.ctime(time.time()).replace(':','_').replace(' ','_')
  if Shopping_List == [] :
    Isotope_List = Isotope_List[679:848] 
    
  else :
    Isotope_List_holder = []
  
    for  i in  Shopping_List:
      Isotope_List_holder.append(Isotope_List[i])
    
    Isotope_List = Isotope_List_holder

  if Unknown_Isotope == False:
      
      if Radioactive_Shopping_List  == False :
          Ordered_Predictions = (-eval_result).argsort()

          Most_Probable_Index_1st = []
          
          Most_Probable_Index_2nd  = []



          for  i in Ordered_Predictions:

              Most_Probable_Index_1st.append(i[0])
              Most_Probable_Index_2nd.append(i[1])

          if Training_Logs == False:
            for MP1 , MP2 , real , prob in zip( Most_Probable_Index_1st , Most_Probable_Index_2nd , df_test_eval , eval_result):
                
                answer = ""
                if MP1 == real :
                    answer = "Correct"
                
                else:
                    answer= "Wrong"

                print("{:20s} : {:.2f}  {:20s} : {:.2f} Answer Given: {:8s} ..... {:11s} ".format(Isotope_List[MP1], prob[MP1], Isotope_List[MP2] , prob[MP2] , answer , Isotope_List[real] ))

          elif Training_Logs == True:
            for MP1 , MP2 , real , prob in zip( Most_Probable_Index_1st , Most_Probable_Index_2nd , df_test_eval , eval_result):
                
                answer = ""
                if MP1 == real :
                    answer = "Correct"
                
                else:
                    answer= "Wrong"

                with open(path+"\Text_Files\Further_Eval_{}.txt".format(Log_Date),'a') as file:
                  file.write( "{:20s} : {:.2f}  {:20s} : {:.2f} Answer Given: {:8s} ..... {:11s} \n".format(Isotope_List[MP1], prob[MP1], Isotope_List[MP2] , prob[MP2] , answer , Isotope_List[real] ) )



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


          for  i in Ordered_Predictions:

              Most_Probable_Index_1st.append(i[0])
              Most_Probable_Index_2nd.append(i[1])

          for MP1 , MP2  , prob in zip( Most_Probable_Index_1st , Most_Probable_Index_2nd  , eval_result):

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

# "Show_Confusion_Matrix()" will display a Confusion Matrix based of the model evaluation.
# "eval_result" a list of numpy arrays of predictions made by the model each position in the array matches to an isotope.
# "Isotope_List" is the list of all isotopes from the CSV file that is returned from "Read_File()".
# "df_test_eval" a list of integers that matched to the indexes of the correct isotopes.
# "Shopping_List" is the list of specific isotopes the user is looking for.


def Show_Confusion_Matrix(eval_result, Isotope_List , df_test_eval=None , Shopping_List = None ):
  if Shopping_List == [] :
    Isotope_List = Isotope_List[679:848] 
    
  else :
    Isotope_List_holder = []

    for  i in  Shopping_List:
      Isotope_List_holder.append(Isotope_List[i])
    
    Isotope_List = Isotope_List_holder
 
  Ordered_Predictions = (-eval_result).argsort()

  Most_Probable_Index_1st = []

  for  i in Ordered_Predictions:

    Most_Probable_Index_1st.append(i[0])
  
  Most_Probable_Index_1st = list(map( lambda i : Isotope_List[i] , Most_Probable_Index_1st))
  df_test_eval = list(map( lambda i : Isotope_List[i] , df_test_eval))


  data = pd.DataFrame( data = {'Actual': df_test_eval , 'Predicted': Most_Probable_Index_1st } )

  confusion_matrix = pd.crosstab(data['Actual'], data['Predicted'], rownames=['Actual'], colnames=['Predicted'], margins = True)

  sn.heatmap(confusion_matrix, annot=True)
  plt.show()



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# "Unknown_Isotope()" will allow the user to see how the program might work in the real world (Only works for decay chain of 0)
# "model" is the trained model.
# "filename" is the file locative of the CSV file containing all of the isotope half-lifes , daughters and types of decay.
# "Unit_Of_Time" is the unit of time decided by the user "Seconds", "Minutes" , "Hours" and "Days".
# "Radioactive_Shopping_List" is "True" if there is a shopping list and "False" if there is no shopping list.
# "Shopping_List" is the list of specific isotopes the user is looking for.


def Unknown_Isotope ( model , filename ,Unit_Of_Time = 'Seconds' , Radioactive_Shopping_List = False , Shopping_List = None ) :

  time_list = [ i for i in range(0,10000)]

  DF , Isotope_List = Read_File( filename = filename , Unit_Of_Time = Unit_Of_Time  )


  Original_Type_Of_Decay= DF.pop('Type_of_Decay')

  Type_Of_Decay =  np.sort ( Original_Type_Of_Decay.unique().tolist() )

  Type_Of_Decay = { index:i for i,index in zip( Type_Of_Decay , range(len(Type_Of_Decay)) )}

  Unknown_Isotope_Dict ={ 'N'           : [],
                          't'           : [],
                          'Decay_Type'  : [] }



  decay_constant = np.log(2) / float(input("Half Life in ({:}) : ".format( Unit_Of_Time )))
  
  for index , isotope  in enumerate(Type_Of_Decay.values()):
    print("{:35s} : ({:}) ".format( isotope, str(index) ) )

  while True:
      
      Decay_Type = input("\ninput type of decay  : ")

      if Decay_Type.isdigit() == True:
        if int(Decay_Type) in Type_Of_Decay.keys() : 
          Decay_Type = int(Decay_Type)
          break

          

  N0 = 1
  for t in time_list:
      
      Ns = Bateman_equation( X0 = N0 , Decay_constants = [decay_constant] , t=t )
      total_abundance = sum( Ns )

      Unknown_Isotope_Dict['N'].append(total_abundance)                          
      Unknown_Isotope_Dict['t'].append(t)
      Unknown_Isotope_Dict['Decay_Type'].append(Decay_Type)

      if total_abundance == 0.001* N0 :
          break

  Unknown_Isotope = pd.DataFrame(Unknown_Isotope_Dict)

  scaler = MinMaxScaler()

  Unknown_Isotope[['t']] = scaler.fit_transform(Unknown_Isotope[['t']])
  Unknown_Isotope['Decay_Type'] =  Unknown_Isotope['Decay_Type'].map( lambda x: x/len(Type_Of_Decay) )

  eval_result  , df_test_eval  = Evaluate( Unknown_Isotope , model , Unknown_Isotope = True)
  if Radioactive_Shopping_List == False:
      Further_Evalutaion (    eval_result = eval_result, Isotope_List = Isotope_List, df_test_eval=df_test_eval , Unknown_Isotope = True ,
                              Radioactive_Shopping_List = False  , Shopping_List = Shopping_List , Training_Logs = False , 
                              path = ""
                                )
  else:  
      Further_Evalutaion (  eval_result, Isotope_List , df_test_eval=df_test_eval , Unknown_Isotope = True , 
                            Radioactive_Shopping_List = True  , Shopping_List = Shopping_List , Training_Logs = False ,
                            path = "")

  print(Unknown_Isotope)


  return 

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# "Progression()" is a funtion that creats a progress bar for the user interface, It takes only two arguments.
# "possision" is the current progress of the task being executed.
# "termination" is the total number of task that have to be executed before the end of the task.

def Progression(possision , termination):
  progress = 100 * (possision / float(termination))
  bar = str('~'*(100-(100-int(progress)))+'→' +' ' * (100-int(progress)))
  print( "\r¦{}¦ {:.0f}%".format(bar ,progress ),end='')
  


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# "Load_DataFrame()" ask the user to input the file path of a dataframe to be used .

def Load_DataFrame():
  DF = pd.read_pickle(input("Enter file name or path : "))

  return DF


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# "Specific_model()" ask the user to input the file path of a Model to be used .


def Specific_model():
  loaded_model = load_model(input("Enter file name or path : "))
  loaded_model.summary()

  return loaded_model

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# "Example_model()" load the example model .


def Example_model(filename):
  loaded_model = load_model(filename)
  loaded_model.summary()

  return loaded_model


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~END OF FUNCTIONS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
