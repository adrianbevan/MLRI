from Code_Functions import *

# This Python script provides a user interface that changes as different options and configurations are chosen # 




path , file_paths , Operating_system = Get_OS()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# "UI_Options" is a dictionary that has values of variables that need to be passed between this script and the "Code_Functions" scirpt.


UI_Options = {  'Training_DataFrame'        : False ,           # "Training_DataFrame" is a check to see the user has generated a Pandas DataFrame for training.
                'Testing_DataFrame'         : False ,           # "Testing_DataFrame" is a check to see the user has generated a Pandas DataFrame for evaluating trained models.
                'Trained_Model'             : False ,           # "Trained_Model" is a check to see the user has a trained or load model to use.
                'Training_Logs'             : False ,           # "Training_Logs" saves each model when a epoch is completed into a file and saves the output of "Further_Evalutaion".
                'Isotope_List'              : False ,           # "Isotope_List" is a list containing all of the names of radioisotopes.
                'Model_Evaluated'           : False ,           # "Model_Evaluated" check to see if a model has been evaluated with a "Testing_DataFrame"
                'Decay_Chain'               : 0     ,           # "Decay_Chain" is how long the decay chain will be for radioisotopes when creating a "Training_DataFrame" or "Testing_DataFrame"
                'Number_Of_Replicants'      : 0     ,           # "Number_Of_Replicants" is the amount of times the creation of a DataFrame is repeated, guassian noise will be applied to half-lifes on each repeat.
                'Radioactive_Shopping_List' : False ,           # "Radioactive_Shopping_List" checks whether specific raidioisotopes to be used have been selected.
                'Shopping_List'             : []    ,           # "Shopping_List" is a list of all the radioisotopes that have been selected.
                'Decay_Type_Seperation'     : False ,           # "Decay_Type_Seperation" seperates the different types of decay when creating DataFrames to give more detail on how radioisotopes decay.
                'Unit_Of_Time'              :'Seconds' }        # "Unit_Of_Time" is the time that the user wants the data to be in "Seconds", "Minutes" , "Hours" and "Days".


refresh()
while True:


    print("{:5s}{:30s} ({:}) \n".format("","Classification Of Nucleides By Half Life" , UI_Options['Unit_Of_Time' ] ))

    print("{:50s} {:50s}".format("Create :","1"))
    print("{:50s} {:50s}".format("Load :","2"))
    
    if UI_Options['Training_DataFrame'] == True  or UI_Options['Trained_Model'] ==True :
        print("{:50s} {:50s}".format("Train Model :","3"))


    print("{:50s} {:50s}".format("Plot Dataframe :","4"))

    if UI_Options['Testing_DataFrame'] == True and UI_Options['Trained_Model'] == True:  
        print("{:50s} {:50s}".format("Evaulate Model :","5"))
    
    if UI_Options['Model_Evaluated'] == True:
        print("{:50s} {:50s}".format("Detailed Evaluation of Model :","6"))

    if UI_Options['Trained_Model'] == True:
        print("{:52s} {:50s}".format("\n\nUnknown Isotope :","7"))

    print("{:50s} {:50s}".format("Configure:","8"))
    print("{:50s} {:50s}".format("Info :","i"))
    print("{:51s} {:50s}".format("\nQuit :","q"))
    
    Option=input("option : ")

    if Option=="1":
        Option  =''
        


        while True:
            refresh()

            print("{:20s}{:30s}".format("","Create \n"))
            print("{:50s} {:50s}".format("Create Training Data :","1"))
            print("{:50s} {:50s}".format("Create Testing Data :","2"))

            if UI_Options['Training_DataFrame'] == True:
                print("{:50s} {}".format("Save Training Data :","3"))

            if UI_Options['Testing_DataFrame'] == True:
                print("{:50s} {:50s}".format("Save Testing Data :","4"))
            
            if UI_Options['Isotope_List'] == True:
                print("{:50s} {:50s}".format("Save Isotope List :","5"))

            if UI_Options['Training_DataFrame'] == True:
                print("{:50s} {}".format("Show Training Data :","6"))

            if UI_Options['Testing_DataFrame'] == True:
                print("{:50s} {}".format("Show Testing Data :","7"))
            


            print("{:51s} {:50s}".format("\nBack :","q"))

            Option=input("option : ")
            
            if Option == '1':
                refresh()
                try:

                    Training_Data, Isotope_List = Read_File( filename = str( path + file_paths['CSV_filename'] ) , Unit_Of_Time = UI_Options['Unit_Of_Time']  )

                except:
                    print("FileNotFoundError: [Errno 2] No such file or directory:{}".format(file_paths['CSV_filename']))
                    print("File Path in varaible 'filename' (line 21) \n")
                    time.sleep(2)
                    continue
                
                Training_Data = Create_Data_Set( DF = Training_Data  , std = 0.005, Num_Of_Replicates = UI_Options['Number_Of_Replicants']  , Unit_Of_Time = UI_Options['Unit_Of_Time'] , decay_chain = UI_Options['Decay_Chain'] , List_Type = 'Long' , Decay_Seperation = UI_Options["Decay_Type_Seperation"] ,Shopping_List = UI_Options['Shopping_List']  )
                
                
                UI_Options['Training_DataFrame']=True
                UI_Options['Isotope_List'] = True

            elif Option == '2':
                refresh()

                try:

                    Testing_Data, Isotope_List = Read_File( filename = str( path + file_paths['CSV_filename']) , Unit_Of_Time = UI_Options['Unit_Of_Time']  )
                  
                except:
                    print("File Path in varaible 'filename' (line 21) \n")
                    print("File Path in varaible 'filename' (line 21) \n")
                    time.sleep(2)
                    continue

                Testing_Data = Create_Data_Set( DF = Testing_Data  , std = 0.005 , Num_Of_Replicates = 0 , Unit_Of_Time = UI_Options['Unit_Of_Time'] , decay_chain= UI_Options['Decay_Chain']  , List_Type = 'Testing' , Decay_Seperation = UI_Options["Decay_Type_Seperation"] , Shopping_List = UI_Options['Shopping_List'] )
                
                
                UI_Options['Testing_DataFrame']=True
                UI_Options['Isotope_List'] = True


            elif Option=='3' and  UI_Options['Training_DataFrame'] == True:
                
                Training_Data.to_pickle(input("Pick filename for Training Data "))
            
            elif Option=='4' and  UI_Options['Testing_DataFrame'] == True :

                Testing_Data.to_pickle(input("Pick filename for Testing Data"))

            elif Option=='5' and  UI_Options['Isotope_List'] == True:

                with open('Isotope_List.txt','w') as file:
                    file_content='\n'.join(Isotope_List)
                    file.write(file_content)

            elif Option=='6' and  UI_Options['Training_DataFrame'] == True :

                refresh()
                while True:
                    print(Training_Data)
                    print("{:51s} {:50s}".format("\nQuit :","q"))

                    Option=input("option : ")
                    if Option == 'q':
                        break

            elif Option=='7' and  UI_Options['Testing_DataFrame'] == True :

                refresh()
                while True:                    
                    print(Testing_Data)
                    print("{:51s} {:50s}".format("\nQuit :","q"))

                    Option=input("option : ")
                    if Option == 'q':
                        break
                        
                        
            elif Option=='8' and  UI_Options['Training_DataFrame'] == True:
                Training_Data.to_csv('Training_Data.csv')
                print('saved')
                time.sleep(4)


            elif Option == 'q':
                break
            
        

    elif Option == "2":
        Option=''
        
        while True:
            refresh()
            
            print("{:20s}{:30s}".format("","Load \n"))
            print("{:50s} {:50s}".format("Load Training Data :","1"))
            print("{:50s} {:50s}".format("Load Testing Data :","2"))
            print("{:50s} {:50s}".format("Load Model :","3"))
            print("{:50s} {:50s}".format("Isotope List :","4"))
            print("{:51s} {:50s}".format("\nBack :","q"))
            Option=input("option : ")

            if Option == '1':
                refresh()

                try:
                    Training_Data = Load_DataFrame()
                    UI_Options['Training_DataFrame']=True
                except:
                    print("Training Data Missing or Path not entered corectly")
                    time.sleep(2)


            elif Option == '2':
                refresh()

                try:
                    Testing_Data = Load_DataFrame()
                    UI_Options['Testing_DataFrame']=True
                except:
                    print("Testing Data Missing or Path not entered corectly")
                    time.sleep(2)



            elif Option == '3':


                while True :
                    refresh()
                    Option=''
                    print("{:20s}{:30s}".format("","Load Model \n"))
                    print("{:50s} {:50s}".format("Load Specific Model :","1"))
                    print("{:51s} {:50s}".format("\nBack :","q"))

                    Option=input("option : ")

                    if Option=='1':

                        clear_output()
                        os.system('cls||clear')
                        time.sleep(0.5)

                        try:
                            model = Specific_model()
                            UI_Options['Trained_Model']=True
                        except:
                            print("Model file Missing or Path not entered corectly")
                            time.sleep(2)

                            
                    elif Option=='q':
                        break
                

            elif Option == '4':
 
                while True :
                    refresh()
                    Option=''
                    print("{:20s}{:30s}".format("","Load Isotpe List \n"))
                    print("{:50s} {:50s}".format("Load Specific Isotope List :","1"))
                    print("{:50s} {:50s}".format("Load Example  Isotope List :","2"))
                    print("{:51s} {:50s}".format("\nBack :","q"))

                    Option=input("option : ")

                    if Option == '1':
                        
                        try:
                            Data = open(input("Enter file name or Path : "), 'r').read()
                            Isotope_List = Data.splitlines()
                            UI_Options['Isotope_List']=True
                        
                        except:
                            print("Isotope List File Missing or Path not entered corectly")
                            time.sleep(2)

                    elif Option =='2':
                        try:
                            Data = open(path+'{}'.format(file_paths["Isotope_List"]), 'r').read()
                            Isotope_List = Data.splitlines()
                            UI_Options['Isotope_List']=True
                        except:
                            print("Isotope List File Missing or Path not entered corectly")
                            time.sleep(2)

                    elif Option == 'q':
                        break
          

            elif Option == 'q':
                break
            

    elif Option == "3":
        Option=''
        if UI_Options['Training_DataFrame'] == True  or UI_Options['Trained_Model'] ==True:
            while True:
                
                refresh()

                print("{:20s}{:30s}".format("","Training Screen \n"))

                if UI_Options['Training_DataFrame'] == True:
                    print("{:50s} {:50s}".format("Train New Model :","1"))

                if  UI_Options['Training_DataFrame'] == True and UI_Options['Trained_Model']== True:
                    print("{:50s} {:50s}".format("Train Existing Model :","2"))

                if UI_Options['Trained_Model'] == True:
                    print("{:50s} {:50s}".format("Model Info :","3"))
                
                if UI_Options['Trained_Model'] == True:
                    print("{:50s} {:50s}".format("Save Model :","4"))

                print("{:51s} {:50s}".format("\nQuit :","q"))

                Option=input("option : ")

                Config_Receipt = {  'Unit_Of_Time'              : UI_Options['Unit_Of_Time'] ,
                                    'Isotope Shopping List'     : UI_Options["Shopping_List"],
                                    'Number_Of_Replicants'      : UI_Options['Number_Of_Replicants'] , 
                                    'Decay_Chain'               : UI_Options['Decay_Chain']  ,
                                    'Decay Type Separation'     : UI_Options["Decay_Type_Seperation"]}

                if Option == '1' and UI_Options['Training_DataFrame'] == True and UI_Options['Isotope_List'] == True :
                    refresh()
                    model , history  = Training_V2 ( Training_df = Training_Data , Config_Receipt = Config_Receipt , Training_Logs = UI_Options["Training_Logs"] , Operating_system = Operating_system)
                    UI_Options['Trained_Model']=True

                elif Option == '2' and UI_Options['Training_DataFrame'] and UI_Options['Trained_Model'] == True and UI_Options['Isotope_List'] == True:
                    refresh()
                    try:
                        model , history  = Training_V2 ( Training_Data ,Isotope_List , New_Model=False , model = model)
                        UI_Options['Trained_Model']=True
                    except:
                        print('Must Create a set of "Training Data" to "Train Existing Model" ')
                        time.sleep(2)
                    

                elif Option == '3' and UI_Options['Trained_Model'] == True:
                    refresh()
                    print(model.summary())
                    print("\nWEIGHTS :\n")
                    print(model.weights)
                    while True:
                        print("{:51s} {:50s}".format("\nBack :","q"))
                        Option=input("option : ")

                        if Option =='q':
                            break

                        clear_output()
                        os.system('cls||clear')

                elif Option == '4' and UI_Options['Trained_Model'] == True :
                    refresh()
                    model.save(input("Pick filename for Trained Model"))

                elif Option == 'q':
                    break
            

    elif Option == "4":
 
        while True:
            refresh()
            print("{:20s}{:30s}".format("","Plot \n"))
            print("{:50s} {:50s}".format("Plot All :","1"))
            print("{:50s} {:50s}".format("Plot Specific Radioisotope :","2"))
            print("{:50s} {:50s}".format("Back :","q"))
            Option=input("option : ")
            
            if Option == '1':

                try:
                    Plot_Data_Frame( filename = str( path + file_paths['CSV_filename'] )  , Unit_Of_Time = UI_Options["Unit_Of_Time"] , decay_chain= UI_Options['Decay_Chain'] , Specific_Radioisotope = False , path = path , Shopping_List = UI_Options["Shopping_List"])

                except:
                    print("ERROR : Either Training Data or Testing Data  is missing")
                    time.sleep(5)

            elif Option == '2':
                    Plot_Data_Frame( filename = str( path + file_paths['CSV_filename'] )  , Unit_Of_Time = UI_Options["Unit_Of_Time"]  , decay_chain= UI_Options['Decay_Chain'] , Specific_Radioisotope = True , path = path , Shopping_List = UI_Options["Shopping_List"])


            elif Option == 'q': 
                break
            


    elif Option == "5" and UI_Options['Testing_DataFrame'] == True and UI_Options['Trained_Model'] == True:
        Option=''
        refresh()
        
        eval_result  , df_test_eval  = Evaluate( Testing_Data , model)
        UI_Options['Testing_DataFrame'] = False
        UI_Options['Model_Evaluated'] = True
        while True:
            print("{:50s} {:50s}".format("\nBack:","q"))
            Option=input("option : ")

            if Option == 'q':
                break

    elif Option == "6" and UI_Options['Model_Evaluated'] == True and UI_Options['Isotope_List'] == True:
        while True:
            refresh()
            print("{:20s}{:30s}".format("","Further_Evalutaion \n"))
            print("{:50s} {:50s}".format("Show All predictions :","1"))
            print("{:50s} {:50s}".format("Show Confusion Matrix :","2"))
            print("{:50s} {:50s}".format("Back :","q"))


            Option=input("option : ")
            if Option == "1":
                Further_Evalutaion ( eval_result, Isotope_List ,df_test_eval , Shopping_List = UI_Options['Shopping_List'] , Training_Logs = UI_Options["Training_Logs"] , path = path)
                
                while True:

                    print("{:50s} {:50s}".format("Back :","q"))
                    Option=input("option : ")                    
                    if Option == "q":
                        refresh()
                        break
                    else:
                        refresh()

            elif Option == "2":
                Show_Confusion_Matrix(eval_result =  eval_result, Isotope_List = Isotope_List , df_test_eval = df_test_eval , Shopping_List = UI_Options['Shopping_List'] )
                while True:
                    print("{:50s} {:50s}".format("Back :","q"))
                    Option=input("option : ")  
                    if Option == "q":
                        refresh()
                        break
    
            elif Option == "q":
                refresh()
                break


    elif Option == "7" and UI_Options['Trained_Model'] == True:
        Option=''
        refresh() 

        Unknown_Isotope( model = model , filename = str(path + file_paths['CSV_filename']) , Unit_Of_Time=UI_Options['Unit_Of_Time'] , Radioactive_Shopping_List= UI_Options["Radioactive_Shopping_List"] , Shopping_List=UI_Options["Shopping_List"] )


        while True:
            print("{:50s} {:50s}".format("Back :","q"))

            Option=input("option : ")

            if Option=="q":

                clear_output()
                os.system('cls||clear')
                time.sleep(0.5)
                break


            else:
                clear_output()
                os.system('cls||clear')
                time.sleep(0.5)




    elif Option == "8":
        Option=''

        while True:
            refresh()
            print("{:10s}{:}( Currently {:} ) \n".format( "", "Select Unit_Of_Time ", UI_Options['Unit_Of_Time'] ))
            print("{:50s} {:50s}".format("Seconds ( 10 Years ) :","1"))
            print("{:50s} {:50s}".format("Minutes ( 600 Years ) :","2"))
            print("{:50s} {:50s}".format("Hours   ( 360 Centuries ) :","3"))
            print("{:50s} {:50s}".format("Days    ( 864 Millennium ):","4"))
            print("\n{:50} {:50s}".format("Set Decay Chain :","5"))
            print("{:50s} {:50s}".format("Apply Noise :","6"))
            print("\n{:50s} {:50s}".format('Set "Seperate Types of Decay" ({:}):'.format( not UI_Options["Decay_Type_Seperation"] ) , "7") )
            print("{:50s} {:50s}".format('Set "Training Logs" ({}):'.format(str( not UI_Options['Training_Logs'] )),"8"))
            print("\n{:50s} {:50s}".format("Select Specific Isotopes:","9"))
            print("{:50s} {:50s}".format('Switch OS from "{:}":'.format( Operating_system ), "10") )
            print("\n{:50s} {:50s}".format("Back :","q"))


            Option=input("option : ")
            if Option == "1":
                UI_Options['Unit_Of_Time'] = "Seconds"
                UI_Options['Training_DataFrame']    = False
                UI_Options['Testing_DataFrame']     = False

            elif Option == "2":
                UI_Options['Unit_Of_Time'] = "Minutes"
                UI_Options['Training_DataFrame']    = False
                UI_Options['Testing_DataFrame']     = False

            elif Option == "3":
                UI_Options['Unit_Of_Time'] = "Hours"
                UI_Options['Training_DataFrame']    = False
                UI_Options['Testing_DataFrame']     = False

            elif Option == "4":
                UI_Options['Unit_Of_Time'] = "Days"
                UI_Options['Training_DataFrame']    = False
                UI_Options['Testing_DataFrame']     = False

            elif Option == "5":
                
                Option = int(UI_Options['Decay_Chain'] )
                               
                while True:
                    refresh()                   
                    print("{:10s}{:}( Currently {:} ) \n".format( "", "Select Decay Chain Length ", UI_Options['Decay_Chain'] ))
                    print("{:50s} {:50s}\n".format("Pick Number between:","0-30"))

                    if UI_Options["Shopping_List"] != []:

                        print("{:50s} {:50s}\n".format("Show Example Decay Chian:","e"))


                    print("{:50s} {:50s}".format("Back:","q"))
                    Option=input("option : ")
                    if Option in [str(i) for i in range(31)]:
                        UI_Options['Decay_Chain'] = int(Option)
                    
                    elif Option=='e' and UI_Options["Shopping_List"] != []:

                        refresh()

                        Display_Decay_Chain(    filename = str( path + file_paths['CSV_filename'] )  , 
                                                Unit_Of_Time = UI_Options['Unit_Of_Time']            , 
                                                chain_num= UI_Options['Decay_Chain']                 ,
                                                Shopping_List = UI_Options["Shopping_List"] )
                        

                    elif Option=='q':
                        break


            elif Option == "6":
                
                while True:
                    refresh() 
                    print("{:10s}{:}( Currently {:} ) \n".format( "", "Select Amount of Replications ", UI_Options['Number_Of_Replicants'] ))
                    print("{:50s} \n".format("Pick Number :"))
                    print("{:50s} {:50s}".format("Back :","q"))
                    Option=input("option : ")

                    try:
                        Option = int(Option)
                        if Option >= 0 : UI_Options['Number_Of_Replicants']  = Option
                    except:
                        pass

                    if Option == 'q' :
                        break


            elif Option =="7":

                UI_Options["Decay_Type_Seperation"] = not UI_Options["Decay_Type_Seperation"]   

            
            elif Option == "8":
                UI_Options['Training_Logs'] = not UI_Options['Training_Logs']

            elif Option =="9":

                refresh()
                Option = ''

                try:

                    Data = open( str(path +file_paths['Isotope_List']) , 'r').read()
                    Isotope_List = Data.splitlines()
                    UI_Options['Isotope_List']=True
                    try :

                        Shopping_List
                        try:
                            Shopping_List = Isotope_Shopping_List(Isotope_List = Isotope_List , Shopping_List = UI_Options['Shopping_List'] , path = path)
                            UI_Options['Shopping_List'] = Shopping_List
                            
                        except:
                            pass
                            
                    except:
                        try:
                            Shopping_List = Isotope_Shopping_List(Isotope_List = Isotope_List , path = path)
                            UI_Options['Shopping_List'] = Shopping_List
                            UI_Options["Radioactive_Shopping_List"] == True

                        except:
                            print("No Isotopes Selected")
                            UI_Options["Radioactive_Shopping_List"] == False
                            time.sleep(3)

                except:
                    print('Isotope List Not Found , Go into "Load" and load an Isotope List. ')
                    time.sleep(10)

            elif Option =="10":

                Option = ''
                while True:
                    refresh()
                    print("{:10s}{:}( Currently {:} ) \n".format( "", "Select OS ", Operating_system))
                    print("{:50s} {:50s}".format("Windows:","1"))
                    print("{:50s} {:50s}".format("Linux:","2"))
                    print("{:50s} {:50s}".format("Mac:","3"))
                    print("{:50s} {:50s}".format("Back:","q"))
                    Option=input("option : ")

                    if Option == "q":
                        break


                    

            elif Option =="q":
                break



    elif Option== "i":
        RAM_Estimate = 124
        if UI_Options["Shopping_List"] == []:
            Amount_Of_Isotopes = 848-679
        else :
            Amount_Of_Isotopes = len(UI_Options["Shopping_List"])

        if UI_Options['Decay_Type_Seperation'] == False:
            RAM_Estimate = (( RAM_Estimate + (36*Amount_Of_Isotopes) ) * 50000)* 8 *(UI_Options['Number_Of_Replicants']+1) + 3
        else:
            RAM_Estimate = (( RAM_Estimate + (36*Amount_Of_Isotopes) ) * 50000)* 8 *7 *(UI_Options['Number_Of_Replicants']+1) + 3

        RAM_Estimate = RAM_Estimate/ 1073741824 
        while True:
            refresh()
            print("{:20s}{:30s}".format("","Current configuration\n"))

            for  key , value in zip( UI_Options.keys() , UI_Options.values() ):
                print("{:50s} {:50s}".format( str(key) , str(value) ))
            
            print("\n{:51s} {:.2f} GB".format("Estimated Amount Of RAM :",RAM_Estimate))
            print("{:51s} {:50s}".format("\nBack :","q"))

        
            Option=input("option : ")

            if Option == "q":
                break


        
    elif Option == "q":
        refresh()
        print("\nExiting\n")
        break

    refresh()




#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~END OF PROGRAM~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
