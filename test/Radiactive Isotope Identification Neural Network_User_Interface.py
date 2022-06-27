from Code_Functions import Read_File
from Code_Functions import Create_Data_Set
from Code_Functions import Plot_Data_Frame
from Code_Functions import training_V2
from Code_Functions import Evaluate
from Code_Functions import Further_Evalutaion
from Code_Functions import Load_DataFrame
from Code_Functions import Load_model

from IPython.display import clear_output

import time
import os





############### Enter the Isotope_Half_Lifes.CSV file path in below ####################################

filename =str(r'Isotope_Half_Lifes.csv') 


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


UI_Options = {  'Training_DataFrame'    : False ,
                'Testing_DataFrame'     : False , 
                'Trained_Model'         : False , 
                'Isotope_List'          : False ,
                'Model_Evaluated'       : False }

clear_output()
os.system('cls||clear')
time.sleep(0.5)                


while True:
    



    print("{:5s}{:30s}".format("","Classification Of Nucleides By Half Life\n"))

    print("{:50s} {:50s}".format("Create :","1"))
    print("{:50s} {:50s}".format("Load :","2"))
    
    if UI_Options['Training_DataFrame'] == True  or UI_Options['Trained_Model'] ==True :
        print("{:50s} {:50s}".format("Train Model :","3"))

    if UI_Options['Training_DataFrame'] == True  or UI_Options['Testing_DataFrame']== True :    
        print("{:50s} {:50s}".format("Plot Dataframe :","4"))

    if UI_Options['Trained_Model'] == True:  
        print("{:50s} {:50s}".format("Evaulate Model :","5"))
    
    if UI_Options['Model_Evaluated'] == True:
        print("{:50s} {:50s}".format("Detailed vaulation of Model (end program):","6"))

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
                    Training_Data, Isotope_List = Read_File( filename )
                except:
                    print("File Path in varaible 'filename' (line 21) \n")
                    time.sleep(2)
                    continue


                Training_Data = Create_Data_Set( Training_Data , std =0.01)
                
                UI_Options['Training_DataFrame']=True
                UI_Options['Isotope_List'] = True


            elif Option == '2':
                clear_output()
                os.system('cls||clear')
                time.sleep(0.5)

                Testing_Data, Isotope_List = Read_File( filename )
                Testing_Data = Create_Data_Set( Testing_Data , std =0.01)

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
            print("{:50s} {:50s}".format("Isotope List :","4"))
            print("{:51s} {:50s}".format("\nQuit :","q"))
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

            elif Option == '3':
                clear_output()
                os.system('cls||clear')
                time.sleep(0.5)
                try:
                    model = Load_model()
                    UI_Options['Trained_Model']=True
                except:
                    print("Model file Missing or Path not entered corectly")
                    time.sleep(2)
            
            elif Option == '4':
                clear_output()
                os.system('cls||clear')
                time.sleep(0.5)
                try:
                    Data = open(input("Enter file name or Path : "), 'r').read()
                    Isotope_List = Data.splitlines()
                    UI_Options['Isotope_List']=True
                
                except:
                    print("Isotope List File Missing or Path not entered corectly")
                    time.sleep(2)
            


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
                    print("{:50s} {:50s}".format("Train Model :","1"))

                if UI_Options['Trained_Model'] == True:
                    print("{:50s} {:50s}".format("Model Info :","2"))
                
                if UI_Options['Trained_Model'] == True:
                    print("{:50s} {:50s}".format("Save Model :","3"))

                print("{:51s} {:50s}".format("\nQuit :","q"))

                Option=input("option : ")

                if Option == '1' and UI_Options['Training_DataFrame'] == True:
                    clear_output()
                    os.system('cls||clear')
                    time.sleep(0.5)
                    model , history , df_train = training_V2 ( Training_Data , Isotope_List )
                    UI_Options['Trained_Model']=True


                elif Option == '2' and UI_Options['Trained_Model'] == True:
                    clear_output()
                    os.system('cls||clear')
                    time.sleep(0.5)
                    print(model.summary())
                    while True:
                        print("{:51s} {:50s}".format("\nQuit :","q"))
                        Option=input("option : ")

                        if Option =='1':
                            clear_output()
                            os.system('cls||clear')
                            time.sleep(0.5)
                            break

                        clear_output()
                        os.system('cls||clear')

                elif Option == '3' and UI_Options['Trained_Model'] == True :
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

    elif Option == "4":

        if UI_Options['Training_DataFrame'] ==True or UI_Options['Testing_DataFrame']== True:
            clear_output()
            os.system('cls||clear')
            time.sleep(0.5)

            Option=''
            try:
                Plot_Data_Frame(Training_Data , Isotope_List)
            except:
                Plot_Data_Frame(Testing_Data , Isotope_List)
            else:
                print("ERROR : Either Training Data or Testing Data  is missing")


    elif Option == "5" and UI_Options['Testing_DataFrame'] == True and UI_Options['Trained_Model'] == True:
        Option=''
        clear_output()
        os.system('cls||clear')
        time.sleep(0.5)
        
        eval_result  , df_test_eval  = Evaluate( Testing_Data , model)
        UI_Options['Model_Evaluated'] = True
        while True:
            print("{:50s} {:50s}".format("\nQuit :","q"))
            Option=input("option : ")

            if Option == 'q':
                clear_output()
                os.system('cls||clear')
                time.sleep(0.5)
                break

    elif Option == "6" and UI_Options['Model_Evaluated'] == True:
        clear_output()
        os.system('cls||clear')
        time.sleep(1)
        Further_Evalutaion ( eval_result, Isotope_List ,df_test_eval)
        break

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

