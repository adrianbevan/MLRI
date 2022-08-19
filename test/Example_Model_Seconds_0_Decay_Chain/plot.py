import matplotlib.pyplot as plt

No_Decay_Chain = [
                0.24669583141803741 , 0.34551867842674255 , 0.3772941827774048 , 0.39561575651168823 ,0.40914255380630493 , 0.4205169081687927 , 0.4302493929862976, 0.4370186924934387 , 0.4436897337436676 ,
                0.4504185616970062 ,0.4555895924568176 , 0.4600284993648529 , 0.4630971848964691 , 0.46708452701568604 , 0.46976882219314575
                    ]

One_Decay_Chain = [
                    0.17452430725097656 , 0.23107777535915375 , 0.24872905015945435 , 0.25923624634742737 , 0.2663826048374176 , 0.2721250355243683 , 0.27612265944480896 , 0.2793945372104645 , 0.2823393642902374 , 0.2848595380783081 ,
                    0.28728583455085754 , 0.28975412249565125 , 0.2914433479309082 , 0.29301196336746216  , 0.2941907048225403
                    ]

Two_Decay_Chain = [
                    0.1681540459394455 , 0.21311545372009277 , 0.22521325945854187 , 0.23245647549629211 , 0.23682376742362976 , 0.24030202627182007 , 0.24370397627353668 , 0.24747119843959808 , 0.25025975704193115 , 0.2527652084827423 , 0.25530290603637695,
                    0.2572804391384125 , 0.2588612139225006  , 0.26039135456085205 , 0.26166629791259766
                    ]
epochs = [i for i in range(len(No_Decay_Chain))]

plt.plot( epochs , No_Decay_Chain  , label = "Decay Chain = 0 ")

plt.plot( epochs , One_Decay_Chain , label = "Decay Chain = 1 ")

plt.plot( epochs , Two_Decay_Chain , label = "Decay Chain = 2 ")
plt.legend()
plt.xlabel("Number Of Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy/Epochs When Training")
plt.show()