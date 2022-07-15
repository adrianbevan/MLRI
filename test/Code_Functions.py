import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tensorflow import keras
from keras.layers.core import Dense
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def Read_File( filename , Daughter_Data = False):
    
    DF = pd.read_csv(filepath_or_buffer = filename)

    Isotope_List = list(DF.pop('Isotope'))

    if Daughter_Data == False:
        DF = DF.drop(columns=['Daughter_1','Daughter_1_Half_Life(Hours)','Daughter_1_Prob'])

    return DF , Isotope_List 


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

    X = X0 * np.e**(-decay_constant*t) 

    return  X 


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#



def Set_Unit_Of_Time( DF , Unit_Of_Time = None):

    Original_Time = list(DF.columns)[0]


    if Original_Time == 'Half_Life(Seconds)':


        if Unit_Of_Time == 'Minutes':

            DF[Original_Time] = DF[Original_Time].map(lambda x : x/60 )

            DF.rename(columns={Original_Time :'Half_Life(Minutes)'} ,   inplace = True)

        elif Unit_Of_Time == 'Hours':

            DF[Original_Time] = DF[Original_Time].map(lambda x : x/3600 )

            DF.rename(columns={Original_Time :'Half_Life(Hours)'}   ,   inplace = True)

        elif Unit_Of_Time == 'Days':

            DF[Original_Time] = DF[Original_Time].map(lambda x : x/(24*3600))

            DF.rename(columns={Original_Time :'Half_Life(Days)'}    ,   inplace = True)


    if Original_Time == 'Half_Life(Minutes)':

        if Unit_Of_Time == 'Seconds':

            DF[Original_Time] = DF[Original_Time].map(lambda x : x*60 )

            DF.rename(columns={Original_Time :'Half_Life(Seconds)'} ,   inplace = True)


        elif Unit_Of_Time == 'Hours':

            DF[Original_Time] = DF[Original_Time].map(lambda x : x/60 )

            DF.rename(columns={Original_Time :'Half_Life(Minutes)'} ,   inplace = True)

        elif Unit_Of_Time == 'Days':

            DF[Original_Time] = DF[Original_Time].map(lambda x : x/(24*60))

            DF.rename(columns={Original_Time :'Half_Life(Days)'}    ,   inplace = True)


        else:   pass


    if Original_Time == 'Half_Life(Hours)':

        if Unit_Of_Time == 'Seconds':

            DF[Original_Time] = DF[Original_Time].map(lambda x : x*3600)

            DF.rename(columns={Original_Time :'Half_Life(Seconds)'} ,   inplace = True)


        elif Unit_Of_Time == 'Minutes':

            DF[Original_Time] = DF[Original_Time].map(lambda x : x*60)

            DF.rename(columns={Original_Time :'Half_Life(Minutes)'} ,   inplace = True)


        elif Unit_Of_Time == 'Days':

            DF[Original_Time] = DF[Original_Time].map(lambda x : x/24)

            DF.rename(columns={Original_Time :'Half_Life(Days)'}    ,   inplace = True)


        else:   pass


    if Original_Time == 'Half_Life(Days)':

        if Unit_Of_Time == 'Seconds':

            DF[Original_Time] = DF[Original_Time].map(lambda x : x/(24*3600) )

            DF.rename(columns={Original_Time :'Half_Life(Seconds)'} ,   inplace = True)

        elif Unit_Of_Time == 'Minutes':

            DF[Original_Time] = DF[Original_Time].map(lambda x : x/(24*60) )

            DF.rename(columns={Original_Time :'Half_Life(Minutes)'} ,   inplace = True)

        elif Unit_Of_Time == 'Hours':

            DF[Original_Time] = DF[Original_Time].map(lambda x : x/24 )

            DF.rename(columns={Original_Time :'Half_Life(Minutes)'} ,   inplace = True)

        else:   pass
     

    return DF


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


def Create_Data_Set( DF , std =0.01 , Num_Of_Replicates = 10 , Unit_Of_Time = 'Seconds'):

    DF = Set_Unit_Of_Time( DF , Unit_Of_Time = Unit_Of_Time)


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


    time_list = [ ]

    for i in range(1000):

        time_list.append( np.random.randint(1,10000) )


    N0 = 1

    for t,zipper in zip(time_list,range(len(time_list))):

        for isotope , index in zip(Decay_const,range(len(Decay_const))):

            for isotope_Decay_constant in isotope :

                
                N=Exp_form( N0 , isotope_Decay_constant , t )
              
                Ap_list['N'].append( N[0] )
                Ap_list['t'].append(t)
                Ap_list['Decay_Type'].append( Type_Of_Decay[str(Original_Type_Of_Decay[index])] )    
                Ap_list['Isotope'].append(index)


        print('{} / {}'.format(zipper, len(time_list)))


    DF = pd.DataFrame(Ap_list)



    return DF



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#



def Plot_Data_Frame(filename , Unit_Of_Time = 'Seconds'):
    
    print("Plotting Data please wait ...")

    DF , Isotope_List = Read_File( filename )

    DF = Set_Unit_Of_Time( DF , Unit_Of_Time = Unit_Of_Time)

    DF = pd.DataFrame(DF['Half_Life({})'.format( Unit_Of_Time )])

    DF['Half_Life({})'.format( Unit_Of_Time )]= DF['Half_Life({})'.format( Unit_Of_Time )].map(lambda x : np.log(2)/x )


    Time_Dict = { i : [ ] for i in range((10**4)+1) }

    N0=1

    for isotope in list(DF.values):

        for t in Time_Dict:

            Time_Dict[t].append(Exp_form( N0 , isotope  , t ))

    Time_Dict = pd.DataFrame(Time_Dict)  



    X_values = list(Time_Dict.columns.values)

    for index, rows in Time_Dict.iterrows():
        # Create list for the current row

        plt.plot(X_values , list(Time_Dict.iloc[index]) )

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
                            #Dense(64, activation='sigmoid'),
                            Dense(len(Isotope_List),activation='softmax')])


        model.compile(optimizer=Adam(learning_rate=0.01), 
                    loss='sparse_categorical_crossentropy', 
                    metrics=['accuracy',])



        history  = model.fit(   scaled_DF, train_labels , 

                                validation_split=0.1 , epochs=25 ,

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


def Evaluate( Testing_df , model , Unknown_Isotope = False):

    df_test_eval = None

    try:
        df_test_eval = Testing_df.pop('Isotope')
        scaled_df_test,df_test_eval = shuffle(Testing_df,df_test_eval)
    
    except:
        scaled_df_test = Testing_df
        pass

    scaler = MinMaxScaler()

    scaled_df_test[['t','Decay_Type']] = scaler.fit_transform(scaled_df_test[['t','Decay_Type']])

    eval_result = model.predict(x=scaled_df_test)

    if Unknown_Isotope == False:

        model.evaluate(x=scaled_df_test , y=df_test_eval , verbose=1 )
        

    return eval_result  , df_test_eval 

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


def Further_Evalutaion ( eval_result, Isotope_List , df_test_eval=None , Unknown_Isotope = False ):


    if Unknown_Isotope == False:

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
            print("{:20s} : {:.2f}  {:20s} : {:.2f}  ".format(Isotope_List[MP1], prob[MP1], Isotope_List[MP2] , prob[MP2] ))


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


def Unknown_Isotope ( model , filename ,Unit_Of_Time = 'Seconds'  ) :

    DF , Isotope_List = Read_File( filename )

    DF = Set_Unit_Of_Time( DF , Unit_Of_Time = Unit_Of_Time)

    Original_Type_Of_Decay= DF.pop('Type_of _Decay_1')

    Type_Of_Decay =  np.sort ( Original_Type_Of_Decay.unique().tolist() )

    Type_Of_Decay = { i:index for i,index in zip( Type_Of_Decay , range(len(Type_Of_Decay)) )}

    Unknown_Isotope_Dict ={ 'N'           : [],
                            't'           : [],
                            'Decay_Type'  : [] }

    time_list = [ i for i in  range(1,10**4)]

    decay_constant = np.log(2) / float(input("Half Life in ({:}) : ".format( Unit_Of_Time )))
    
    print(Type_Of_Decay) 
    while True:
        
        Decay_Type = input("input type of decay (word)")

        if Decay_Type in Type_Of_Decay.keys():

            break

    N0 = 1
    for t in time_list:

        Unknown_Isotope_Dict['N'].append(Exp_form( N0 ,decay_constant , t ))
        Unknown_Isotope_Dict['t'].append(t)
        Unknown_Isotope_Dict['Decay_Type'].append( (Type_Of_Decay[Decay_Type])/(len(Type_Of_Decay)-1) )


        if Exp_form( N0 ,decay_constant , t ) <= 0.01*N0:
            break

    Unknown_Isotope = pd.DataFrame(Unknown_Isotope_Dict)

 
    eval_result  , df_test_eval  = Evaluate( Unknown_Isotope , model , Unknown_Isotope = True)

    Further_Evalutaion ( eval_result =eval_result , Isotope_List =Isotope_List , df_test_eval=df_test_eval , Unknown_Isotope = True )

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


def Test_func(DF , Option = None , Parent_List=None):
    
    
    if Option == 1:

        graph_num = int(input("Input number between (1,{}) :".format(len(DF))))

        try:
            CDf=DF[graph_num-1:graph_num]
            try:
                P_Dc=np.log(2)/float(CDf['Half_Life(Seconds)'])
                D_DC=np.log(2)/float(CDf['Daughter_1_Half_Life(Seconds)'])
            except:
                pass
            N0=5000
            P_x=[]
            D_x=[]
            t=[]

            j=0

            while j<1000000:
                P_x.append( N0*np.e**(-P_Dc*j) )

                D_x.append(
                    N0*( P_Dc/(D_DC - P_Dc) )*( (np.e**(-P_Dc*j)) -(np.e**(-D_DC*j)))
                )
                t.append(j)

                if N0*( P_Dc/(D_DC - P_Dc) )*( (np.e**(-P_Dc*j)) -(np.e**(-D_DC*j)))<10  and N0*np.e**(-P_Dc*j)< 10:
                    break

                j+=0.1

            y=[]

            for n,m in zip(P_x,D_x):
                y.append(n+m)

            plt.plot(t,P_x , label = str("Parent_"+Parent_List[graph_num-1]) )
            plt.plot(t , D_x , label = CDf['Daughter_1'] )
            plt.plot(t,y)
            plt.legend()
            plt.show()
            
        except:
            print("Error")





#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~END OF FUNCTIONS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
