
from Code_Functions import *











############### Enter the file path below if not working ####################################





path , file_paths , Operating_system = Get_OS()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


UI_Options = {  'Training_DataFrame'    : False ,
                'Testing_DataFrame'     : False , 
                'Trained_Model'         : False , 
                'Isotope_List'          : False ,
                'Model_Evaluated'       : False ,
                'Daughter_Data'         : False ,
                'Radioactive_Shopping_List' : False , 
                'Unit Of Time'          :'Seconds' }

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

    print("{:50s} {:50s}".format("Testing :","8"))
    print("{:50s} {:50s}".format("Configure:","9"))
    print("{:50s} {:50s}".format("Info :","i"))
    print("{:51s} {:50s}".format("\nQuit :","q"))
    
    Option=input("option : ")

    if Option=="1":
        Option=''
        refresh()

        while True:

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
                    if UI_Options["Daughter_Data"]== False:
                        Training_Data, Isotope_List = Read_File( filename= str(path + file_paths['CSV_filename']) , Unit_Of_Time = UI_Options['Unit Of Time'] )
                    else:
                        Training_Data, Isotope_List = Read_File( filename = str( path + file_paths['CSV_filename'] ) , Daughter_Data= True , Unit_Of_Time = UI_Options['Unit Of Time']  )

                except:
                    print("FileNotFoundError: [Errno 2] No such file or directory:{}".format(file_paths['CSV_filename']))
                    print("File Path in varaible 'filename' (line 21) \n")
                    time.sleep(2)
                    continue
                
                if UI_Options['Daughter_Data'] == False:

                    Training_Data = Create_Data_Set( Training_Data , std =0.01 , Num_Of_Replicates = 10 , Unit_Of_Time = UI_Options['Unit Of Time'] )

                else:
                    Training_Data = Create_Data_Set_2( Training_Data , std =0.01 , Num_Of_Replicates = 10 , Unit_Of_Time = UI_Options['Unit Of Time'] )

                UI_Options['Training_DataFrame']=True
                UI_Options['Isotope_List'] = True

            elif Option == '2':
                refresh()

                try:
                    if UI_Options["Daughter_Data"]== False:
                        Testing_Data, Isotope_List = Read_File( filename = str( path + file_paths['CSV_filename']) , Unit_Of_Time = UI_Options['Unit Of Time']  )
                    else :
                        Testing_Data, Isotope_List = Read_File( filename = str( path + file_paths['CSV_filename'] ) , Daughter_Data = True , Unit_Of_Time = UI_Options['Unit Of Time'] ) 
                except:
                    print("File Path in varaible 'filename' (line 21) \n")
                    time.sleep(2)
                    continue

                if UI_Options['Daughter_Data'] == False:

                    Testing_Data = Create_Data_Set( Testing_Data , std =0.01 , Num_Of_Replicates = 1 , Unit_Of_Time = UI_Options['Unit Of Time'] )

                else:
                    Testing_Data = Create_Data_Set_2( Testing_Data , std =0.01 , Num_Of_Replicates = 1 , Unit_Of_Time = UI_Options['Unit Of Time'] )


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


            elif Option == 'q':
                refresh()

                break
            
            refresh()
        

    elif Option == "2":
        Option=''
        refresh()

        while True:


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
                            model = Load_model()
                            UI_Options['Trained_Model']=True
                        except:
                            print("Model file Missing or Path not entered corectly")
                            time.sleep(2)

                    elif Option == '2':

                        while True :
                            refresh()

                            Option=''
                            print('{:10s}{:}(Currrently on "{}" & Daughter_Data "{}")\n'.format("","Load Example Model \n", UI_Options['Unit Of Time'] , UI_Options["Daughter_Data"] ) )
                            print("{:50s} {:50s}".format("Load Seconds  Model :","1"))
                            print("{:50s} {:50s}".format("Load Minutes  Model :","2"))
                            print("{:50s} {:50s}".format("Load Hours    Model :","3"))
                            print("{:50s} {:50s}".format("Load Days     Model :","4"))

                            if UI_Options["Daughter_Data"] == True :
                                print("\n{:50s} {:50s}".format("Load Seconds  Model for Daughter Data :","5"))
                                print("{:50s} {:50s}".format("Load Minutes  Model for Daughter Data :","6"))
                                print("{:50s} {:50s}".format("Load Hours    Model for Daughter Data :","7"))
                                print("{:50s} {:50s}".format("Load Days     Model for Daughter Data :","8"))

                            print("{:51s} {:50s}".format("\nBack :","q"))
                            Option=input("option : ")

                            if Option=='1':

                                try:
                                    model = Load_Example_model(filename = str( path + file_paths['Example_Seconds'] ))
                                    UI_Options['Trained_Model']=True
                                except:
                                    print("Example file(s) are missing ")
                                    time.sleep(2)

                            elif Option=='2':

                                try:
                                    model = Load_Example_model(filename = str( path + file_paths['Example_Minutes'] ))
                                    UI_Options['Trained_Model']=True
                                except:
                                    print("Example file(s) are missing ")
                                    time.sleep(2)


                            elif Option=='3':

                                try:
                                    model = Load_Example_model(filename = str( path + file_paths['Example_Hours'] ))
                                    UI_Options['Trained_Model']=True
                                except:
                                    print("Example file(s) are missing ")
                                    time.sleep(2)
                                    

                            elif Option=='4':

                                try:
                                    model = Load_Example_model(filename = str( path + file_paths['Example_Days'] ))
                                    UI_Options['Trained_Model']=True
                                except:
                                    print("Example file(s) are missing ")
                                    time.sleep(2)

                            elif Option=='5':

                                try:
                                    model = Load_Example_model(filename = str( path + file_paths['Example_Seconds_DD'] ))
                                    UI_Options['Trained_Model']=True
                                except:
                                    print("Example file(s) are missing ")
                                    time.sleep(2)

                            elif Option=='q':
                                refresh()
                                break
                    elif Option=='q':
                        refresh()
                        break
                

            elif Option == '4':

                refresh()                     

                while True :
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
                        refresh()
                        break

                    refresh()

          

            elif Option == 'q':
                refresh()
                break
            
            refresh()


    elif Option == "3":
        Option=''
        if UI_Options['Training_DataFrame'] == True  or UI_Options['Trained_Model'] ==True:
            while True:

                clear_output()
                os.system('cls||clear')
                time.sleep(0.5)

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

                if Option == '1' and UI_Options['Training_DataFrame'] == True and UI_Options['Isotope_List'] == True :
                    refresh()
                    model , history  = training_V2 ( Training_Data , Isotope_List )
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
                            refresh()
                            break

                        clear_output()
                        os.system('cls||clear')

                elif Option == '4' and UI_Options['Trained_Model'] == True :
                    refresh()
                    model.save(input("Pick filename for Trained Model"))

                elif Option == 'q':
                    refresh()
                    break
            
                refresh()

        else:
            refresh()

    elif Option == "4":


        refresh()

        Option=''
        try:
            Plot_Data_Frame( filename = str(path+ file_paths["CSV_filename"]) , Unit_Of_Time = UI_Options['Unit Of Time'] , Daughter_Data = UI_Options["Daughter_Data"])
        except:
            print("ERROR : Either Training Data or Testing Data  is missing")
            time.sleep(5)

        refresh()


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
                refresh()
                break

    elif Option == "6" and UI_Options['Model_Evaluated'] == True and UI_Options['Isotope_List'] == True:
        refresh()
        Further_Evalutaion ( eval_result, Isotope_List ,df_test_eval ,)
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

    elif Option== "8":
        Option=''
        refresh()
        DF, Isotope_List = Read_File(filename = str( path + file_paths['CSV_filename'] ) , Daughter_Data= True , Unit_Of_Time = UI_Options['Unit Of Time']  )

        
        while True:
            print("{:20s}{:30s}".format("","Testing (Probably not working) \n"))
            print("{:50s} {:50s}".format("Print Decay graph :","1"))
            print("{:50s} {:50s}".format("- :","2"))
            print("{:50s} {:50s}".format("- :","3"))
            print("{:50s} {:50s}".format("Back :","q"))


            Option=input("option : ")
            if Option == "1":
                print(DF)
                Test_func(DF , Option = 1 , Parent_List= Isotope_List , Unit_Of_Time = UI_Options['Unit Of Time'] )


            elif Option =="q":
                refresh()
                break

            refresh()


    elif Option == "9":
        Option=''
        refresh()

        while True:
            print("{:10s}{:}( Currently {:} ) \n".format( "", "Select Unit of time ", UI_Options['Unit Of Time'] ))
            print("{:50s} {:50s}".format("Seconds:","1"))
            print("{:50s} {:50s}".format("Minutes:","2"))
            print("{:50s} {:50s}".format("Hours:","3"))
            print("{:50s} {:50s}".format("Days:","4"))
            print("{:51s} {:50s}".format("\nSet Daughter Data ({}):".format(str( not UI_Options['Daughter_Data'] )),"5"))
            print("{:50s} {:50s}".format("Select Specific Isotopes:","6"))
            print("{:50s} {:50s}".format('Switch OS from "{:}":'.format(Operating_system),"7"))
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
                UI_Options['Daughter_Data'] = not UI_Options['Daughter_Data']

            elif Option =="6":

                refresh()
                Option = ''
                if UI_Options["Isotope_List"] == False :  
                    print("Isotope List Not Found")
                    time.sleep(3)

                else:
                    try :
                        Shopping_List
                        try:
                            Shopping_List = Isotope_Shopping_List(Isotope_List = Isotope_List, Shopping_List = Shopping_List)
                            
                        except:
                            pass
                            
                    except:
                        try:
                            Shopping_List = Isotope_Shopping_List(Isotope_List = Isotope_List )
                            UI_Options["Radioactive_Shopping_List"] == True

                        except:
                            print("No Isotopes Selected")
                            UI_Options["Radioactive_Shopping_List"] == False
                            time.sleep(3)

            elif Option =="7":
                refresh()
                Option = ''
                while True:
                    print("{:10s}{:}( Currently {:} ) \n".format( "", "Select OS ", Operating_system))
                    print("{:50s} {:50s}".format("Windows:","1"))
                    print("{:50s} {:50s}".format("Linux:","2"))
                    print("{:50s} {:50s}".format("Back:","q"))
                    Option=input("option : ")

                    if Option == "q":
                        break

                    refresh()

                    

            elif Option =="q":
                refresh()
                break

            refresh()
       


    elif Option== "i":
        refresh()
        print(str(UI_Options)+"\n")


        
    elif Option == "q":
        clear_output()
        os.system('cls||clear')
        print("\nExiting\n")
        break
        


    else:
        refresh()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~END OF PROGRAM~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
