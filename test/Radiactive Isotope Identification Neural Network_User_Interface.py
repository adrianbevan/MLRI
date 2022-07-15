import os
import time

from IPython.display import clear_output

from Code_Functions import *







############### Enter the file path below if not working ####################################




file_path = os.path.dirname(__file__)

filename ='\Isotope_Half_Lifes.csv'


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


UI_Options = {  'Training_DataFrame'    : False ,
                'Testing_DataFrame'     : False , 
                'Trained_Model'         : False , 
                'Isotope_List'          : False ,
                'Model_Evaluated'       : False ,
                'Unit Of Time'          :'Seconds' }

clear_output()
os.system('cls||clear')
time.sleep(0.5)                


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
    print("{:50s} {:50s}".format("Change Unit of time:","9"))
    print("{:50s} {:50s}".format("Info :","i"))
    print("{:51s} {:50s}".format("\nQuit :","q"))
    
    Option=input("option : ")

    if Option=="1":
        Option=''
        clear_output()
        os.system('cls||clear')
        time.sleep(0.5)

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


            print("{:51s} {:50s}".format("\nBack :","q"))

            Option=input("option : ")
            
            if Option == '1':
                clear_output()
                os.system('cls||clear')
                time.sleep(0.5)
                try:
                    Training_Data, Isotope_List = Read_File( str(file_path+ filename) ) 
                except:
                    print("FileNotFoundError: [Errno 2] No such file or directory:{}".format(filename))
                    print("File Path in varaible 'filename' (line 21) \n")
                    time.sleep(2)
                    continue

                Training_Data = Create_Data_Set( Training_Data , std =0.01 , Num_Of_Replicates = 10 , Unit_Of_Time = UI_Options['Unit Of Time'] )

                UI_Options['Training_DataFrame']=True
                UI_Options['Isotope_List'] = True


            elif Option == '2':
                clear_output()
                os.system('cls||clear')
                time.sleep(0.5)

                try:
                    Testing_Data, Isotope_List = Read_File( str(file_path+ filename) )
                    
                except:
                    print("File Path in varaible 'filename' (line 21) \n")
                    time.sleep(2)
                    continue

                Testing_Data = Create_Data_Set( Testing_Data , std =0.01 , Num_Of_Replicates = 1 , Unit_Of_Time = UI_Options['Unit Of Time'] )

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


            elif Option == 'q':
                clear_output()
                os.system('cls||clear')
                time.sleep(0.5)

                break
            
            clear_output()
            os.system('cls||clear')
            time.sleep(0.5)
        

    elif Option == "2":
        Option=''
        clear_output()
        os.system('cls||clear')
        time.sleep(0.5)

        while True:


            print("{:20s}{:30s}".format("","Load \n"))
            print("{:50s} {:50s}".format("Load Training Data :","1"))
            print("{:50s} {:50s}".format("Load Testing Data :","2"))
            print("{:50s} {:50s}".format("Load Model :","3"))
            print("{:51s} {:50s}".format("\nBack :","q"))
            Option=input("option : ")

            if Option == '1':
                clear_output()
                os.system('cls||clear')
                time.sleep(0.5)

                try:
                    Training_Data = Load_DataFrame()
                    UI_Options['Training_DataFrame']=True
                except:
                    print("Training Data Missing or Path not entered corectly")
                    time.sleep(2)

                try:
                    Data = open(input("Enter file name or Path : "), 'r').read()
                    Isotope_List = Data.splitlines()
                    UI_Options['Isotope_List']=True
                
                except:
                    print("Isotope List File Missing or Path not entered corectly")
                    time.sleep(2)

            elif Option == '2':
                clear_output()
                os.system('cls||clear')
                time.sleep(0.5)

                try:
                    Testing_Data = Load_DataFrame()
                    UI_Options['Testing_DataFrame']=True
                except:
                    print("Testing Data Missing or Path not entered corectly")
                    time.sleep(2)

                try:
                    Data = open(input("Enter file name or Path : "), 'r').read()
                    Isotope_List = Data.splitlines()
                    UI_Options['Isotope_List']=True
                
                except:
                    print("Isotope List File Missing or Path not entered corectly")
                    time.sleep(2)

            elif Option == '3':


                while True :
                    clear_output()
                    os.system('cls||clear')
                    time.sleep(0.5)
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

                            clear_output()
                            os.system('cls||clear')
                            time.sleep(0.5)

                            Option=''
                            print('{:20s}{:30s}(Currrently on "{}")'.format("","Load Example Model \n", UI_Options['Unit Of Time']) )
                            print("{:50s} {:50s}".format("Load Seconds  Model :","1"))
                            print("{:50s} {:50s}".format("Load Minutes  Model :","2"))
                            print("{:50s} {:50s}".format("Load Hours    Model :","3"))
                            print("{:50s} {:50s}".format("Load Days     Model :","4"))
                            print("{:51s} {:50s}".format("\nBack :","q"))
                            Option=input("option : ")

                            if Option=='1':

                                # try:
                                model = Load_Example_model(filename = str(file_path+"\Example_Models\Model_Half_Life(Seconds)_001"))
                                UI_Options['Trained_Model']=True
                                # except:
                                #     print("Example file(s) are missing ")
                                #     time.sleep(2)

                            elif Option=='2':

                                try:
                                    model = Load_Example_model(filename = str(file_path+"\Example_Models\Model_Half_Life(Minutes)_001"))
                                    UI_Options['Trained_Model']=True
                                except:
                                    print("Example file(s) are missing ")
                                    time.sleep(2)


                            elif Option=='3':

                                try:
                                    model = Load_Example_model(filename = str(file_path+"\Example_Models\Model_Half_Life(Hours)_001"))
                                    UI_Options['Trained_Model']=True
                                except:
                                    print("Example file(s) are missing ")
                                    time.sleep(2)
                                    

                            elif Option=='4':

                                try:
                                    model = Load_Example_model(filename = str(file_path+"\Example_Models\Model_Half_Life(Days)_001"))
                                    UI_Options['Trained_Model']=True
                                except:
                                    print("Example file(s) are missing ")
                                    time.sleep(2)

                            elif Option=='q':
                                clear_output()
                                os.system('cls||clear')
                                time.sleep(0.5)
                                break


                    elif Option == 'q':

                        clear_output()
                        os.system('cls||clear')
                        time.sleep(0.5)
                        break


           

            elif Option == 'q':
                clear_output()
                os.system('cls||clear')
                time.sleep(0.5)
                break
            
            clear_output()
            os.system('cls||clear')
            time.sleep(0.5)


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

                if Option == '1' and UI_Options['Training_DataFrame'] == True :
                    clear_output()
                    os.system('cls||clear')
                    time.sleep(0.5)
                    model , history  = training_V2 ( Training_Data , Isotope_List )
                    UI_Options['Trained_Model']=True

                elif Option == '2' and UI_Options['Training_DataFrame'] and UI_Options['Trained_Model'] == True:
                    clear_output()
                    os.system('cls||clear')
                    time.sleep(0.5)
                    try:
                        model , history  = training_V2 ( Training_Data ,Isotope_List , New_Model=False , model = model)
                        UI_Options['Trained_Model']=True
                    except:
                        print('Must Create a set of "Training Data" to "Train Existing Model" ')
                        time.sleep(2)
                    

                elif Option == '3' and UI_Options['Trained_Model'] == True:
                    clear_output()
                    os.system('cls||clear')
                    time.sleep(0.5)
                    print(model.summary())
                    print("\nWEIGHTS :\n")
                    print(model.weights)
                    while True:
                        print("{:51s} {:50s}".format("\nBack :","q"))
                        Option=input("option : ")

                        if Option =='q':
                            clear_output()
                            os.system('cls||clear')
                            time.sleep(0.5)
                            break

                        clear_output()
                        os.system('cls||clear')

                elif Option == '4' and UI_Options['Trained_Model'] == True :
                    clear_output()
                    os.system('cls||clear')
                    time.sleep(0.5)
                    model.save(input("Pick filename for Trained Model"))

                elif Option == 'q':
                    clear_output()
                    os.system('cls||clear')
                    time.sleep(0.5)
                    break
            
            clear_output()
            os.system('cls||clear')
            time.sleep(0.5)

        else:
            clear_output()
            os.system('cls||clear')
            time.sleep(0.5)

    elif Option == "4":


        clear_output()
        os.system('cls||clear')
        time.sleep(0.5)

        Option=''
        try:
            Plot_Data_Frame( filename = str(file_path+ filename) , Unit_Of_Time = UI_Options['Unit Of Time'] )
        except:
            print("ERROR : Either Training Data or Testing Data  is missing")
            time.sleep(5)


        clear_output()
        os.system('cls||clear')
        time.sleep(0.5)

    elif Option == "5" and UI_Options['Testing_DataFrame'] == True and UI_Options['Trained_Model'] == True:
        Option=''
        clear_output()
        os.system('cls||clear')
        time.sleep(0.5)
        
        eval_result  , df_test_eval  = Evaluate( Testing_Data , model)
        UI_Options['Testing_DataFrame'] = False
        UI_Options['Model_Evaluated'] = True
        while True:
            print("{:50s} {:50s}".format("\nBack:","q"))
            Option=input("option : ")

            if Option == 'q':
                clear_output()
                os.system('cls||clear')
                time.sleep(0.5)
                break

    elif Option == "6" and UI_Options['Model_Evaluated'] == True:
        clear_output()
        os.system('cls||clear')
        time.sleep(0.5)
        Further_Evalutaion ( eval_result, Isotope_List ,df_test_eval)
        break


    elif Option == "7" and UI_Options['Trained_Model'] == True:
        Option=''
        clear_output()
        os.system('cls||clear')
        time.sleep(0.5)
        Unknown_Isotope( model , filename = str(file_path+ filename) , Unit_Of_Time=UI_Options['Unit Of Time'] )

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


    elif Option== "8":
        Option=''
        clear_output()
        os.system('cls||clear')
        time.sleep(1)
        DF, Isotope_List = Read_File(filename = str(file_path+ filename) , Daughter_Data= True )
        while True:
            print("{:20s}{:30s}".format("","Testing (Probably not working) \n"))
            print("{:50s} {:50s}".format("Print Decay graph :","1"))
            print("{:50s} {:50s}".format("- :","2"))
            print("{:50s} {:50s}".format("- :","3"))
            print("{:50s} {:50s}".format("Back :","q"))


            Option=input("option : ")
            if Option == "1":
                print(DF)
                Test_func(DF , Option = 1 , Parent_List= Isotope_List)


            elif Option =="q":
                clear_output()
                os.system('cls||clear')
                time.sleep(1)
                break

            clear_output()
            os.system('cls||clear')
            time.sleep(1)


    elif Option == "9":
        Option=''
        clear_output()
        os.system('cls||clear')
        time.sleep(0.5)

        while True:
            print("{:20s}{:30s} ( Currently {} ) \n".format( "", "Select Unit of time ", UI_Options['Unit Of Time'] ))
            print("{:50s} {:50s}".format("Seconds:","1"))
            print("{:50s} {:50s}".format("Minutes:","2"))
            print("{:50s} {:50s}".format("Hours:","3"))
            print("{:50s} {:50s}".format("Days:","4"))
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


            elif Option =="q":
                clear_output()
                os.system('cls||clear')
                time.sleep(0.5)
                break

            clear_output()
            os.system('cls||clear')
            time.sleep(0.5)


    elif Option== "i":
        clear_output()
        os.system('cls||clear')
        time.sleep(1)
        print(str(UI_Options)+"\n")


        
    elif Option == "q":
        clear_output()
        os.system('cls||clear')
        print("\nExiting\n")
        break
        


    else:
        clear_output()
        os.system('cls||clear')
        time.sleep(0.5)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~END OF PROGRAM~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
