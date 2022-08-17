
from Code_Functions import *











############### Enter the file path below if not working ####################################





path , file_paths , Operating_system = Get_OS()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


UI_Options = {  'Training_DataFrame'        : False ,
                'Testing_DataFrame'         : False , 
                'Trained_Model'             : False ,
                'Training_Logs'             : False , 
                'Isotope_List'              : False ,
                'Model_Evaluated'           : False ,
                'Decay_Chain'               : 0     ,
                'Number_Of_Replicants'      : 0     ,
                'Radioactive_Shopping_List' : False ,
                'Shopping_List'             : [] , 
                'Unit Of Time'              :'Seconds' }


refresh()
while True:


    print("{:5s}{:30s} ({:}) \n".format("","Classification Of Nucleides By Half Life" , UI_Options['Unit Of Time' ] ))

    print("{:50s} {:50s}".format("Create :","1"))
    print("{:50s} {:50s}".format("Load :","2"))
    
    if UI_Options['Training_DataFrame'] == True  or UI_Options['Trained_Model'] ==True :
        print("{:50s} {:50s}".format("Train Model :","3"))


    print("{:50s} {:50s}".format("Plot Dataframe :","4"))

    if UI_Options['Testing_DataFrame'] == True and UI_Options['Trained_Model'] == True:  
        print("{:50s} {:50s}".format("Evaulate Model :","5"))
    
    if UI_Options['Model_Evaluated'] == True:
        print("{:50s} {:50s}".format("Detailed vaulation of Model (end program):","6"))

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

                    Training_Data, Isotope_List = Read_File( filename = str( path + file_paths['CSV_filename'] ) , Unit_Of_Time = UI_Options['Unit Of Time']  )

                except:
                    print("FileNotFoundError: [Errno 2] No such file or directory:{}".format(file_paths['CSV_filename']))
                    print("File Path in varaible 'filename' (line 21) \n")
                    time.sleep(2)
                    continue
                
                Training_Data = Create_Data_Set( DF = Training_Data  , std =0.01 , Num_Of_Replicates = UI_Options['Number_Of_Replicants']  , Unit_Of_Time = UI_Options['Unit Of Time'] , decay_chain = UI_Options['Decay_Chain'] , List_Type = 'Long' , Shopping_List = UI_Options['Shopping_List']  )
                
                
                UI_Options['Training_DataFrame']=True
                UI_Options['Isotope_List'] = True

            elif Option == '2':
                refresh()

                try:

                    Testing_Data, Isotope_List = Read_File( filename = str( path + file_paths['CSV_filename']) , Unit_Of_Time = UI_Options['Unit Of Time']  )
                  
                except:
                    print("File Path in varaible 'filename' (line 21) \n")
                    print("File Path in varaible 'filename' (line 21) \n")
                    time.sleep(2)
                    continue

                Testing_Data = Create_Data_Set( DF = Testing_Data  , std =0.01 , Num_Of_Replicates = 0 , Unit_Of_Time = UI_Options['Unit Of Time'] , decay_chain=0 , List_Type = 'Random' , Shopping_List = UI_Options['Shopping_List'])
                
                
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
                    print("{:50s} {:50s}".format("Load Example  Model :","2"))
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

                    elif Option == '2':

                        while True :
                            refresh()

                            Option=''
                            print('{:10s}{:}(Currrently on "{}" & Decay_Chain "{}")\n'.format("","Load Example Model \n", UI_Options['Unit Of Time'] , UI_Options["Decay_Chain"] ) )
                            print("{:50s} {:50s}".format("Load Seconds  Model (0 Decay Chain):","1"))
                            print("{:50s} {:50s}".format("Info :","i"))
                            print("{:51s} {:50s}".format("\nBack :","q"))
                            Option=input("option : ")

                            if Option=='1':

                                try:
                                    model = Example_model(filename = str( path + file_paths['Example_Seconds'] ))
                                    UI_Options['Trained_Model']=True
                                except:
                                    print("Example file(s) are missing ")
                                    time.sleep(2)


                            elif Option=='i':
                                print(path)
                                print(file_paths)
                                time.sleep(30)
                            elif Option=='q':
                                break
                            
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
                            Data = open(path+'\Isotope_List.txt', 'r').read()
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

                Config_Receipt = {  'Unit Of Time'              : UI_Options['Unit Of Time'] ,
                                    'Number_Of_Replicants'      : UI_Options['Number_Of_Replicants'] , 
                                    'Decay_Chain'               : UI_Options['Decay_Chain']  }

                if Option == '1' and UI_Options['Training_DataFrame'] == True and UI_Options['Isotope_List'] == True :
                    refresh()
                    model , history  = training_V2 ( Training_df = Training_Data , Config_Receipt = Config_Receipt , Training_Logs = UI_Options["Training_Logs"] , Operating_system = Operating_system)
                    UI_Options['Trained_Model']=True

                elif Option == '2' and UI_Options['Training_DataFrame'] and UI_Options['Trained_Model'] == True and UI_Options['Isotope_List'] == True:
                    refresh()
                    try:
                        model , history  = training_V2 ( Training_Data ,Isotope_List , New_Model=False , model = model)
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
                    Plot_Data_Frame( filename = str( path + file_paths['CSV_filename'] )  , Unit_Of_Time = UI_Options["Unit Of Time"] , decay_chain= UI_Options['Decay_Chain'] , Specific_Radioisotope = False)

                except:
                    print("ERROR : Either Training Data or Testing Data  is missing")
                    time.sleep(5)

            elif Option == '2':
                    Plot_Data_Frame( filename = str( path + file_paths['CSV_filename'] )  , Unit_Of_Time = UI_Options["Unit Of Time"]  , decay_chain= UI_Options['Decay_Chain'] , Specific_Radioisotope = True)


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
        refresh()
        Further_Evalutaion ( eval_result, Isotope_List ,df_test_eval , Shopping_List = UI_Options['Shopping_List'] )
        break


    elif Option == "7" and UI_Options['Trained_Model'] == True:
        Option=''
        refresh() 
        try:
            if UI_Options["Radioactive_Shopping_List"] == True :
                Unknown_Isotope(    model , filename = str(path + file_paths['CSV_filename']) , Unit_Of_Time=UI_Options['Unit Of Time'] ,

                                    Radioactive_Shopping_List = UI_Options["Radioactive_Shopping_List"] , Shopping_List = Shopping_List )

            else:
                Unknown_Isotope( model , filename = str(path + file_paths['CSV_filename']) , Unit_Of_Time=UI_Options['Unit Of Time']  )


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
        except:
            print("ERROR")



    elif Option == "8":
        Option=''

        while True:
            refresh()
            print("{:10s}{:}( Currently {:} ) \n".format( "", "Select Unit of time ", UI_Options['Unit Of Time'] ))
            print("{:50s} {:50s}".format("Seconds:","1"))
            print("{:50s} {:50s}".format("Minutes:","2"))
            print("{:50s} {:50s}".format("Hours:","3"))
            print("{:50s} {:50s}".format("Days:","4"))
            print("\n{:50} {:50s}".format("Set Decay Chain :","5"))
            print("{:50s} {:50s}".format("Apply Gaussian Noise :","6"))
            print("{:50s} {:50s}".format("Set Training Logs ({}):".format(str( not UI_Options['Training_Logs'] )),"7"))
            print("\n{:50s} {:50s}".format("Select Specific Isotopes:","8"))
            print("{:50s} {:50s}".format('Switch OS from "{:}":'.format(Operating_system),"9"))
            print("{:50s} {:50s}".format("Back :","q"))


            Option=input("option : ")
            if Option == "1":
                UI_Options['Unit Of Time'] = "Seconds"
                UI_Options['Training_DataFrame']    = False
                UI_Options['Testing_DataFrame']     = False

            elif Option == "2":
                UI_Options['Unit Of Time'] = "Minutes"
                UI_Options['Training_DataFrame']    = False
                UI_Options['Testing_DataFrame']     = False

            elif Option == "3":
                UI_Options['Unit Of Time'] = "Hours"
                UI_Options['Training_DataFrame']    = False
                UI_Options['Testing_DataFrame']     = False

            elif Option == "4":
                UI_Options['Unit Of Time'] = "Days"
                UI_Options['Training_DataFrame']    = False
                UI_Options['Testing_DataFrame']     = False

            elif Option == "5":
                
                Option = int(UI_Options['Decay_Chain'] )
                Example_chain = ['nobelium-256', 'fermium-252', 'californium-248', 'curium-244', 'plutonium-240', 'uranium-236', 'thorium-232', 'radium-228', 'actinium-228', 'thorium-228', 'radium-224', 'radon-220', 'polonium-216', 'lead-212', 'bismuth-212', 'polonium-212', 'lead-208']
                
                while True:
                    refresh()                   
                    print("{:10s}{:}( Currently {:} ) \n".format( "", "Select Decay Chain Length ", UI_Options['Decay_Chain'] ))
                    print("{:50s} {:50s}\n".format("Pick Number between:","0-30"))
                    print("{:50s} {:50s}\n".format("Example : ",'e'))
                    
                    Example_chain_str=""
                    for isotope in range(int (UI_Options['Decay_Chain'] )+1 ):
                        try:
                            if isotope>3:
                                Example_chain_str = Example_chain_str +'...=>{:}'.format(Example_chain[int(Option)])
                                break
                            Example_chain_str = Example_chain_str +'=>{:}'.format(Example_chain[isotope])
                            
                        except:
                            break

                    print("{:50s}\n".format(Example_chain_str))
                    print("{:50s} {:50s}".format("Back:","q"))
                    Option=input("option : ")
                    if Option in [str(i) for i in range(31)]:
                        UI_Options['Decay_Chain'] = int(Option)
                    
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



            
            elif Option == "7":
                UI_Options['Training_Logs'] = not UI_Options['Training_Logs']

            elif Option =="8":

                refresh()
                Option = ''

                try:

                    Data = open( str(path +file_paths['Isotope_List']) , 'r').read()
                    Isotope_List = Data.splitlines()
                    UI_Options['Isotope_List']=True
                    try :

                        Shopping_List
                        try:
                            Shopping_List = Isotope_Shopping_List(Isotope_List = Isotope_List , Shopping_List = UI_Options['Shopping_List'])
                            UI_Options['Shopping_List'] = Shopping_List
                            
                        except:
                            pass
                            
                    except:
                        try:
                            Shopping_List = Isotope_Shopping_List(Isotope_List = Isotope_List )
                            UI_Options['Shopping_List'] = Shopping_List
                            UI_Options["Radioactive_Shopping_List"] == True

                        except:
                            print("No Isotopes Selected")
                            UI_Options["Radioactive_Shopping_List"] == False
                            time.sleep(3)

                except:
                    print('Isotope List Not Found , Go into "Load" and load an Isotope List. ')
                    time.sleep(10)

            elif Option =="9":

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
        refresh()
        print(str(UI_Options)+"\n")
        continue


        
    elif Option == "q":
        refresh()
        print("\nExiting\n")
        break

    refresh()




#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~END OF PROGRAM~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
