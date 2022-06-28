import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt 
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from tensorflow import keras
from keras.models import Sequential
from keras.layers.core import  Dense
from keras.optimizers  import Adam
from keras.models import load_model

    




#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#



def Read_File( filename ):
    
    DF = pd.read_csv(filename)

    Isotope_List = list(DF.pop('Isotope'))

    return DF  , Isotope_List 



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#




#This Function adds gaussian noise to a set of data when creatiing the digital twin.

def gaussian_noise( Data , mu = 0.0 , std = 0.05 ):

    noise = np.random.normal( mu , std, size = Data.shape)

    Noisy_Data = Data + noise

    return Noisy_Data 




#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


#This function adds guassian noise to the data which is used to create a digital twin.

#'Data' is the input data that the noise is going to be apply to , 'mu' is the mean , and 'std' is standard deviation.  

def Exp_form( X0 ,decay_constant , t ):

    X = X0 * math.e**(-decay_constant*t) 

    return  X 




#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#



#This funtion calculates the the change of X as a function of time using the exponetial decay formula

#DF is the dataframe and std is the standered deviation on the degital twin data currently 1% (0.01) change if you want to.

def Create_Data_Set( DF , std =0.01):

    print('Creating Digital Twin please wait ... ')   

    X0 = 5000

    # Original_Half_Life  is just an array of the original Half-Life in days before the Digital twin is created

    Original_Half_Life = np.array( ( DF['Half_Life(Hours)'] ))
    
 


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#



#input the number the times you want the training data to be replicated with guassian noise (currently set to 1000) Total number of rows = 1000* (Oringinal amount of rows )

#The higher the range the more RAM this dataframe will take up most machines will be able to handle 1000, change at your on discretion.


    for i in range(1000): 

        
        Noisey_Half_Life = gaussian_noise(Original_Half_Life, std=std)        

        Noisey_Decay = ( np.log(2) ) / Noisey_Half_Life

        Noisey_Data = pd.DataFrame( {'Half_Life(Hours)' : Noisey_Half_Life , 'Decay_Constant' : Noisey_Decay } )

        DF = pd.concat([DF, Noisey_Data] )


    
    DF.reset_index(inplace = True, drop = True)
 

    index_start=0

    index_end= len(Original_Half_Life) 

    Decay_data = list(DF['Decay_Constant'])

   
 
#The code below 

    test_xaxis = []
    test_xaxis_loop=0
    multiplyer = 0.0001

    while test_xaxis_loop<=10000000000:
        for i in range(100):
            test_xaxis.append(test_xaxis_loop)
            test_xaxis_loop+=multiplyer
        multiplyer=multiplyer*10 

#The for loop simulates the dacay of each element and appends it to DF

    for  t in test_xaxis:
        

        Moving_Data=[ Exp_form ( X0 , decay_constant , t  ) for decay_constant in Decay_data]

       
        interval_data = pd.DataFrame({ str(t) : Moving_Data  })
        

        DF = pd.concat([ DF , interval_data ],axis=1 )


        index_start +=len( Original_Half_Life )
        index_end+=len( Original_Half_Life )
    

#Makes the list of radioactive isotope names and replaces them with a list of numberical values corresponding to the name of the Isotopes        

    
    Numeric_Isotope_Identifications=[]



    for  i in range( int( len(DF)/len(Original_Half_Life) ) ):

        [Numeric_Isotope_Identifications.append(j) for j in range(len( Original_Half_Life ))]

        
    Numeric_Isotope_Identifications = pd.DataFrame({ 'Isotope' : Numeric_Isotope_Identifications } )

   
   
    
    DF = pd.concat( [DF , Numeric_Isotope_Identifications] , axis=1 )

    DF.pop('Decay_Constant')
    DF.pop('Half_Life(Hours)')

    return DF 

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#





def Plot_Data_Frame(Training_df , Isotope_List):
    
    print("Plotting Data please wait ...")

    try:
        Training_df.pop('Isotope')
    except:
        Training_df = Training_df

    

    X_values = list(Training_df.columns.values)

    for index, rows in Training_df[:len(Isotope_List)].iterrows():
        # Create list for the current row

        plt.plot(X_values , list(Training_df.iloc[index]) )

    plt.show()
    plt.close('all')



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


def training_V2 ( Training_df  ,Isotope_List , New_Model=True , model = None):

    if New_Model== True: 
        sx = MinMaxScaler()
        sy = MinMaxScaler()
        scaled_df = sx.fit_transform(Training_df.drop('Isotope', axis='columns' ))
        
        df_train = Training_df.pop('Isotope')

        Training_df,df_train = shuffle( Training_df, df_train )

        scaler = MinMaxScaler(feature_range=(0,1))

        scaled_df = scaler.fit_transform(Training_df)


        tf.random.set_seed(42)
        
        model = Sequential([Dense(128 ,activation='sigmoid'),
                            Dense(64 , activation='sigmoid'),
                            Dense(len(Isotope_List),activation='softmax')])


        model.compile(optimizer=Adam(learning_rate=0.0001), 
                    loss='sparse_categorical_crossentropy', 
                    metrics=['accuracy',])
        
        history  = model.fit(scaled_df, df_train, epochs=50 ,verbose=2)  #100 looks good 
        

        return model , history ,df_train

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

        return model , history ,df_train


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#




def Evaluate( Testing_df , model):


    df_test_eval = Testing_df.pop('Isotope')
    Testing_df,df_test_eval = shuffle(Testing_df,df_test_eval)

    scaler = MinMaxScaler(feature_range=(0,1))

    scaled_df_test = scaler.fit_transform(Testing_df)

    eval_result = model.predict(x=scaled_df_test)

    rounded_predictions = np.argmax(eval_result,axis=-1)


    cnt=0
    for  i,j in zip(rounded_predictions,df_test_eval):
        if i ==j :
            cnt+=1
    print("Accuracy On Test Set : "+str( cnt/len(rounded_predictions) ) )

    return eval_result  , df_test_eval 

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#



def Further_Evalutaion ( eval_result, Isotope_List ,df_test_eval):

    Ordered_Predictions = (-eval_result).argsort()

    Most_Probable_Index_1st = []
    
    Most_Probable_Index_2nd  = []

    for  i in Ordered_Predictions:

        Most_Probable_Index_1st.append(i[0])
        Most_Probable_Index_2nd.append(i[1])


    for MP1 , MP2 , real , prob in zip( Most_Probable_Index_1st , Most_Probable_Index_2nd , df_test_eval , eval_result):
        
        answer = ""
        if MP1 == real :
            answer = "Correct"
        
        else:
            answer= "Wrong"

        print("{:15s} : {:.2f}  {:15s} : {:.2f} Answer Given: {:} ..... {:11s} ".format(Isotope_List[MP1], prob[MP1], Isotope_List[MP2] , prob[MP2] , answer , Isotope_List[real] ))


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#





def Load_DataFrame():
    DF = pd.read_pickle(input("Enter file name or path : "))

    return DF


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#



def Load_model():
    loaded_model = load_model(input("Enter file name or path : "))
    loaded_model.summary()

    return loaded_model


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~END OF FUNCTIONS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
