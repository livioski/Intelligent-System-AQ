import tensorflow as tf
import random
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy as np
import math
#from Scientific.Functions.Interpolation import InterpolatingFunction

def truncate(value, digits) -> list:
    
    stepper = 10.0 ** digits
    if type(value).__module__ == 'numpy':
        value = value.tolist()
        trunVec = []
        for point in value:
            rounded_point = []
            for i in range(len(point)):
                rounded_point.append(math.trunc(stepper * point[i]) / stepper)
            trunVec.append(rounded_point)
        return trunVec
    if isinstance(value,list):
        trunVec = []
        for i in range(len(value)):
            trunVec.append(math.trunc(stepper * value[i]) / stepper)
        return trunVec
    elif isinstance(value,float):
        return math.trunc(stepper * value) / stepper

def gen_struc_in_out(lines):

    structured_input = []
    structured_output_ang = []
    structured_output_dis = []

    

    #dal file (dalle lines) mi popolo le tre liste
    for line in lines:

        first_split = 0
        second_split = -1
        comma = 0
        rest_of_line = []

        pixel = []
        comp = []

        for char in line:



            if second_split == -1:
                first_split += 1

            else:
                second_split += 1


            if char == "," and comma == 0:
                comma += 1

            elif char == "," and comma == 1:

                pixel = line[2:first_split-2]
                pixel = list(pixel.split(",")) 
                pixel = [float(item) for item in pixel]

                #print(f"Il pixel è {pixel}")

                rest_of_line = line[first_split+1:-1]
                second_split += 1
                
                comma += 1
            
            elif char == "," and comma == 2:
                
                comma += 1

            elif char ==',' and comma == 3:
                comp = rest_of_line[1:second_split-3]
                comp =  list(comp.split(","))
                comp = [float(item) for item in comp]

                coord = rest_of_line[second_split+1:-1]
                coord = list(coord.split(",")) 
                coord = [float(item) for item in coord]

                coord_ang = coord[0]
                coord_dis = coord[1]


                #DALL'ANGOLO CHE SI CREA TRA UN ROBOT E UNALTRO CALCOLO L'ANGOLO CHE SI CREA TRA IL NORD DI UN ROBOT E L'ALTRO
                if coord_ang > 0:
                    if coord_ang > 1.5:
                        coord_ang = -(coord_ang-1.5)
                    elif coord_ang < 1.5:
                        coord_ang = 1.5 - coord_ang
                    elif coord_ang == 1.5:
                        coord_ang = 0
                elif coord_ang < 0:
                    if coord_ang > -1.5:
                        coord_ang = 1.5 - coord_ang 
                    elif coord_ang < -1.5:
                        coord_ang = 0 - (3.1 + coord_ang + 1.5)
                else: 
                    coord_ang = 1.5

                structured_output_ang.append(coord_ang)
                structured_output_dis.append(coord_dis)

                comma += 1

        structured_output_ang = truncate(structured_output_ang,1)
        coup_pix_com = [pixel[0],pixel[1],comp[0],comp[1]]
        structured_input.append(coup_pix_com)

   
    '''it = 0

    #faccio un po di magie con un dizionario per evitare che ci siano 2 input (keypoint e bussola) uguali con diverso output (ang,dis)
    new_dic = {}

    for inp in structured_input:
        new_dic[f"{inp}"] = [structured_output_ang[it],structured_output_dis[it]]
        it += 1

    #e poi dal dizionario mi rifaccio le liste (perchè in input alla nn e alla spline ci vanno le liste)
    structured_input = []
    structured_output_ang = []
    structured_output_dis = []

    for key in new_dic:
        new_key = key[1:-1]
        li = list(new_key.split(","))
        new_key = [float(item) for item in li]

        structured_input.append(new_key)
        structured_output_ang.append(new_dic[key][0])
        structured_output_dis.append(new_dic[key][1])'''

    return structured_input, structured_output_ang, structured_output_dis

def calc_mean_sq(y,y_bar):
    
    summation = 0  #variable to store the summation of differences
    n = len(y) #finding total number of items in list
    for i in range (0,n):  #looping through each element of the list
        if y[i] < 100:
            difference = y[i] - y_bar[i]  #finding the difference between observed and predicted value
            squared_difference = difference**2  #taking square of the differene 
            summation = summation + squared_difference  #taking a sum of all the differences
    MSE = summation/n  #dividing summation by total values to obtain average
    return MSE

def gen_and_train_nn(X_train,Y_train):
    model = tf.keras.models.Sequential([   
    tf.keras.layers.Dense(300, activation='relu'),
    tf.keras.layers.Dense(80, activation='relu'),
    tf.keras.layers.Dense(80, activation='relu'),    
    tf.keras.layers.Dense(80, activation='relu'),
    tf.keras.layers.Dense(80, activation='relu'),
    tf.keras.layers.Dense(500, activation='relu'),
    tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer="Adam", loss="mse")

    model.fit(X_train, Y_train, epochs=400)

    return model

def gen_test_train(inputD,outputD):
    copy_inputD = inputD.copy()
    copy_outputD = outputD.copy()
    X_test = []
    Y_test = []

    picked = 0

    #prendo il 10% del tutto come test, il resto è train
    while picked < len(inputD)/10 :

        index = random.randint(0, len(copy_inputD)-1)
        
        X_test.append(copy_inputD[index])    
        Y_test.append(copy_outputD[index])
        copy_inputD.pop(index)
        copy_outputD.pop(index)
        
        picked += 1

    X_train = copy_inputD
    Y_train = copy_outputD
    

    return X_test, Y_test, X_train, Y_train

ang_path = "model_ang"
dis_path = "model_dis"

with open('dataset.txt') as f:
    lines = f.read().splitlines()

structured_input, structured_output_ang, structured_output_dis = gen_struc_in_out(lines)
print(structured_output_ang)

X_test_ang, Y_test_ang, X_train_ang, Y_train_ang = gen_test_train(structured_input,structured_output_ang)

#X_test_dis, Y_test_dis, X_train_dis, Y_train_dis = gen_test_train(structured_input,structured_output_dis)
'''
x_k = np.arange(0,500,0.1)
y_k = np.arange(0,500,0.1)
x_n = np.arange(0,4,0.1)
y_n = np.arange(0,4,0.1)
data = np.random.rand(len(lats)*len(lons)*len(alts)*len(time)).reshape((len(lats),len(lons),len(alts),len(time)))

axes = (x_k, y_k, x_n, y_n)
f = InterpolatingFunction(axes, data)'''

'''
points = X_train_ang
values = Y_train_ang

grid_x, grid_y = np.mgrid[0:500:0.1, 0:500:0.1]
grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')

plt.imshow(grid_z.T, extent=(0,500,0,500), origin='lower')
plt.show()



grid_z0 = RegularGridInterpolator(points, values)
Y_pred = []
for test in X_test_ang:
    pred = grid_z0(test)
    Y_pred.append(pred)

x, y = np.asarray(points).T
X = np.linspace(0, 500)
Y = np.linspace(0, 500)
X, Y = np.meshgrid(X, Y)
Z = grid_z0(X, Y)
plt.pcolormesh(X, Y, Z, shading='auto')
plt.plot(x, y, "ok", label="input point",  ms=1)
plt.legend()
plt.colorbar()
plt.axis("equal")
plt.show()'''


#model_ang = tf.keras.models.load_model('ang_model')
#model_dis = tf.keras.models.load_model(dis_path)

model_ang = gen_and_train_nn(X_train_ang, Y_train_ang)
#model_dis = gen_and_train_nn(X_train_dis, Y_train_dis)

results_ang = model_ang.predict(X_test_ang)
#results_dis = model_dis.predict(X_test_dis)


'''for i in range(len(results_ang)):
    print(f"per {Y_test_ang[i]} la predizione è invece {results_ang[i]}")'''
mse_ang_nn = calc_mean_sq(results_ang,Y_test_ang)
print(mse_ang_nn)

'''mse_dis_nn = calc_mean_sq(results_dis,Y_test_dis)
print(mse_dis_nn)'''
#mse_dis_sp = calc_mean_sq(Y_pred,Y_test_dis)
#print(f"la splineh a un errore di {mse_dis_sp}")

model_ang.save('ang_model_to_N')
#model_dis.save('dis_model')

