from controller import Supervisor, Camera, Robot, DistanceSensor, Compass
import tensorflow as tf
import sys, math
import cv2
import numpy as np
import random 
import time



def initialize_sensors_motors():

    right_sensor = DistanceSensor('right_dis_sensor')
    left_sensor = DistanceSensor('left_dis_sensor')
    DistanceSensor.enable(right_sensor,16)
    DistanceSensor.enable(left_sensor,16)

    # get the motor devices
    leftMotor = supervisor.getDevice('left wheel motor')
    rightMotor = supervisor.getDevice('right wheel motor')

    return right_sensor,left_sensor,leftMotor,rightMotor

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

def get_robot_rel_coord(trans_field,trans_field_tracked):

    values = trans_field.getSFVec3f()
    values = truncate(values, 1)
    
    values_tracked = trans_field_tracked.getSFVec3f()
    values_tracked = truncate(values_tracked, 1)
    
    
    angle = truncate(math.atan2(values_tracked[2]-values[2], values_tracked[0]-values[0]),1)
    
    distance = truncate(math.sqrt(((values_tracked[0]-values[0])**2)+((values_tracked[2]-values[2])**2) ),1)
    
    
    coord = [angle,distance]

    return coord
    
def setup_detector():
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 0
    params.maxThreshold = 30

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 20
    params.maxArea = 10000


    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.3

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.9

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.0001

    # Create a detector with the parameters
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3 :
            
        detector = cv2.SimpleBlobDetector(params)
    else : 
        
        detector = cv2.SimpleBlobDetector_create(params)

    return detector

def map_ang_to_N(ang):

    if ang > 0:
        if ang > 1.5:
            ang = -(ang-1.5)
        elif ang < 1.5:
            ang = 1.5 - ang
        elif ang == 1.5:
            ang = 0
    elif ang < 0:
        if ang > -1.5:
            ang = 1.5 - ang
        elif ang < -1.5:
            ang = 0 - (3.1 + ang + 1.5)
    else: 
        ang = 1.5
    return ang

def save_img(keypoint,coord):
    pt_str_x = str(keypoint[0]).split(".")
    pt_str_x = str(pt_str_x[0])+","+str(pt_str_x[1])
    pt_str_y = str(keypoint[1]).split(".")
    pt_str_y = str(pt_str_y[0])+","+str(pt_str_y[1])
    pt_str = pt_str_x + "_" + pt_str_y

    coord_ang = str(coord[0]).split(".")
    coord_ang = str(coord_ang[0])+","+str(coord_ang[1])
    coord_dis = str(coord[1]).split(".")
    coord_dis = str(coord_dis[0])+","+str(coord_dis[1])
    coord_str = coord_ang + "_" + coord_dis

    img_name = pt_str + "=" + coord_str + ".png"
    cam.saveImage(img_name, 100)

def avoid_obstacle_routine(right_sensor,left_sensor,changing,aligning,reached_x):
    
    if changing > 0:
        
        changing = changing - 1

    else:
        
        if right_sensor.getValue() < 999:
            reached_x = False

            #set changing
            changing = 30
            #go left
            leftMotor.setVelocity(random.randint(5,15))
            rightMotor.setVelocity(0)

        elif left_sensor.getValue() < 999:
            reached_x = False

            #set changing
            changing = 30
            #go right
            leftMotor.setVelocity(0)
            rightMotor.setVelocity(random.randint(5,15))

        #if no dis_sensor triggered
        else:
            #if aligning skip
            if aligning == True:
                pass
            #else move random
            else:
                leftMotor.setVelocity(random.randint(5,15))
                rightMotor.setVelocity(random.randint(5,15))

    return changing,reached_x

def chase_routine(goal_ang,motor_comp,dist,direction):

    if direction == "to":
        pass
    if direction == "from":
        if goal_ang > 0:
            goal_ang = goal_ang - 3.1
        elif goal_ang < 0:
            goal_ang = goal_ang + 3.1
        else:
            goal_ang = 3.1

    if dist <= 0.6:
        print("fermi che andiamo a sbatte")
        leftMotor.setVelocity(0)
        rightMotor.setVelocity(0)
    else:
        if motor_comp < 0:
            print("+++++++++++++++++++++++++++++++++++++++++++++++")
            #se sono "vicini" i valori, allora tira dritto
            if motor_comp - 0.1 <= goal_ang <= motor_comp + 0.1:
                print("dritto")
                leftMotor.setVelocity(10)
                rightMotor.setVelocity(10)   
            else:
                if goal_ang < 0:  
                    if motor_comp - 0.1 < goal_ang:
                        print("giro sulla sin")
                        leftMotor.setVelocity(8)
                        rightMotor.setVelocity(0)
                    else:
                        print("giro sulla des")
                        leftMotor.setVelocity(0)
                        rightMotor.setVelocity(8)
                else:
                    print("giro sulla des")
                    leftMotor.setVelocity(0)
                    rightMotor.setVelocity(8)
            print("+++++++++++++++++++++++++++++++++++++++++++++++")

        else:
            print("-----------------------------------------------")
            #se sono "vicini" i valori, allora tira dritto
            if motor_comp - 0.1 <= goal_ang <= motor_comp + 0.1:
                print("dritto")
                leftMotor.setVelocity(5)
                rightMotor.setVelocity(5)
            else:
                if goal_ang > 0:  
                    if motor_comp - 0.1 < goal_ang:
                        print("giro sulla sin")
                        leftMotor.setVelocity(8)
                        rightMotor.setVelocity(0)
                    else:
                        print("giro sulla des")
                        leftMotor.setVelocity(0)
                        rightMotor.setVelocity(8)
                else:
                    print("giro sulla sin")
                    leftMotor.setVelocity(8)
                    rightMotor.setVelocity(0)
            print("-----------------------------------------------")

def triangle(type_of_triangle,pred_angle,motor_comp,pred_dis,reached_x,aligning):

    if type_of_triangle == "Equilatero":

        aligning = True

        side = 1
        #x_target = math.sin(1.04) * side
        x_target = 0.5

        print(f"Devo andare a {x_target}")
        x_act = abs(math.sin(pred_angle) * pred_dis)
        print(f"Sto a {x_act}")
        print(f"-----------------------------")
        #se sto sulla sinistra rispetto al vertice alto (nord)

        '''if x_act > x_target + 0.1 or x_act < x_target - 0.1:
            reached_x = False'''

        #reached_x = False
        if pred_angle < 0:
            if reached_x == False:

                #print("mi allineo sulle x")
                if x_act > x_target + 0.1:
                    #per muovermi sulla destra uso il chase, perchè comunque mi va verso il robot, che è alla mia destra
                    #chase_routine(rightMotor,leftMotor,pred_angle,motor_comp,pred_dis,"to")
                    align_x(pred_angle,motor_comp,"right")
                elif x_act < x_target - 0.1:
                    align_x(pred_angle,motor_comp,"left")
                    #chase_routine(rightMotor,leftMotor,pred_angle,motor_comp,pred_dis,"from")
                else:
                    #print("ok sono allineato sulle x")
                    reached_x = True
            else:        
                
                if pred_dis <= side:
                    leftMotor.setVelocity(0)
                    rightMotor.setVelocity(0)
                else:
                    #qui semplifico e mi muovo solo a nord, non sull'asse y come faccio per l'align x
                    #TODO muoversi sull'asse y
                    #print("Mi allineo sulle y")
                    align_nord(side,pred_dis,motor_comp)

        elif pred_angle > 0:
            if reached_x == False:

                print("mi allineo sulle x")
                if x_act < x_target - 0.1:
                    #per muovermi sulla destra uso il chase, perchè comunque mi va verso il robot, che è alla mia destra
                    #chase_routine(pred_angle,motor_comp,pred_dis,"to")
                    align_x(pred_angle,motor_comp,"right")
                elif x_act > x_target + 0.1:
                    align_x(pred_angle,motor_comp,"left")
                    #chase_routine(pred_angle,motor_comp,pred_dis,"from")
                else:
                    print("ok sono allineato sulle x")
                    reached_x = True
            else:        
                
                if pred_dis <= side:
                    leftMotor.setVelocity(0)
                    rightMotor.setVelocity(0)
                else:
                    #qui semplifico e mi muovo solo a nord, non sull'asse y come faccio per l'align x
                    #TODO muoversi sull'asse y
                    print("Mi allineo sulle y")
                    align_nord(side,pred_dis,motor_comp)

    if type_of_triangle == "LineH":
        side = 1
        #x_target = math.sin(1.04) * side
        x_target = 1
    

        print(f"Devo andare a {x_target}")
        x_act = abs(math.sin(pred_angle) * pred_dis)
        print(f"Sto a {x_act}")
        print(f"-----------------------------")
        #se sto sulla sinistra rispetto al vertice alto (nord)

        if x_act > x_target + 0.1 or x_act < x_target - 0.1:
            reached_x = False

        #reached_x = False
        if pred_angle < 0:
            if reached_x == False:

                print("mi allineo sulle x")
                if x_act > x_target + 0.2:
                    #per muovermi sulla destra uso il chase, perchè comunque mi va verso il robot, che è alla mia destra
                    #chase_routine(rightMotor,leftMotor,pred_angle,motor_comp,pred_dis,"to")
                    align_x(pred_angle,motor_comp,"right")
                elif x_act < x_target - 0.1:
                    align_x(pred_angle,motor_comp,"left")
                    #chase_routine(rightMotor,leftMotor,pred_angle,motor_comp,pred_dis,"from")
                else:
                    print("ok sono allineato sulle x")
                    reached_x = True
            else:        
                
                if pred_dis == side - 0.1:
                    leftMotor.setVelocity(0)
                    rightMotor.setVelocity(0)
                elif pred_dis >= side:
                    #qui semplifico e mi muovo solo a nord, non sull'asse y come faccio per l'align x
                    #TODO muoversi sull'asse y
                    print("Mi allineo sulle y")
                    align_nord(side,pred_dis,motor_comp)
                else:
                    leftMotor.setVelocity(0)
                    rightMotor.setVelocity(0)

        elif pred_angle > 0:
            if x_act > x_target + 0.1:
                chase_routine(pred_angle,motor_comp,pred_dis,"to")
            elif x_act < x_target - 0.1:
                chase_routine(pred_angle,motor_comp,pred_dis,"from")
    
    print("------------------")
    return reached_x, aligning

def align_x(pred_angle,motor_comp,direction):

    if direction == "left":
        
    
        if 1.6 >= motor_comp and motor_comp >= 1.4:
            leftMotor.setVelocity(10)
            rightMotor.setVelocity(10)
        else:
            #gira verso sx                
            if (1.4 > motor_comp and  motor_comp >= 0) or (0 >= motor_comp and motor_comp > -1.4):
                
                leftMotor.setVelocity(10)
                rightMotor.setVelocity(0)
            #gira verso dx   
            else:
                
                leftMotor.setVelocity(0)
                rightMotor.setVelocity(10)
    else:
       
        if -1.4 >= motor_comp and motor_comp >= -1.6:
            leftMotor.setVelocity(10)
            rightMotor.setVelocity(10)
        else:
            #gira verso dx
            if (1.4 > motor_comp and motor_comp >= 0) or (0 >= motor_comp and motor_comp > -1.4):
                
                leftMotor.setVelocity(0)
                rightMotor.setVelocity(10)
            #gira verso sx   
            else:
                
                leftMotor.setVelocity(10)
                rightMotor.setVelocity(0)

def align_nord(side,pred_dis,motor_comp):
    
    if motor_comp < 0:
        leftMotor.setVelocity(10)
        rightMotor.setVelocity(0)
    elif motor_comp > 0:
        leftMotor.setVelocity(0)
        rightMotor.setVelocity(10)
    else:
        acc_prop(side,pred_dis)

def acc_prop(target,act):
    error = act - target
    speed = 2
    speed = int(speed + (error*10))
    leftMotor.setVelocity(speed)
    rightMotor.setVelocity(speed)



TIME_STEP = 16

typeOfTriangle = "Equilatero"

supervisor = Supervisor()

cam = Camera("CAM2")
cam.enable(16)

comp = Compass("compass2")
comp.enable(16)

aligning = False
reached_x = False

detector = setup_detector()

# prendo la pos. del robot con la camera installata (dove sta runnando lo script)...
robot_node = supervisor.getFromDef("MY_ROBOT2")
trans_field = robot_node.getField("translation")

right_sensor, left_sensor, leftMotor,rightMotor = initialize_sensors_motors()
leftMotor.setPosition(float('+inf'))
rightMotor.setPosition(float('+inf'))

# ...e la pos (sul simulatore) dell'altro(i) robot(s) per "id"
tracked_node = supervisor.getFromDef("Tracked")
trans_field_tracked = tracked_node.getField("translation")

#mi serve solo per non salvare la prima img "nera"
counter = 1

#changing per la routine di avoid_obst
changing = 0
capture = True


model_ang = tf.keras.models.load_model('ang_model_to_N')
model_dis = tf.keras.models.load_model('dis_model')

while supervisor.step(TIME_STEP) != -1:


    comp_vals = truncate(comp.getValues(),1)
    comp_vals.pop(1)
  
    ang_comp = truncate(math.atan2(comp_vals[1],comp_vals[0]),1)
 

    if counter == 1:
        counter = counter + 1
        key_coord = []
        
    else:
        counter = counter + 1
        # elaboro l'img una volta ogni 7 frame

        if changing == 0:
            if (counter % 5) == 0:

                #print(f"l'angolo dalla bussola è {ang_comp}")
                if ang_comp < 0:
                    motor_comp = ang_comp + 3.1
                elif ang_comp > 0:
                    motor_comp = ang_comp - 3.1
                else:
                    motor_comp = 3.1

                #print(F"Quindi l'angolo tra il nord e motori è {motor_comp}")
                #catturo l'immagine..
                img = cam.getImageArray()
                img = np.float32(img)
                # ..la faccio in grayscale..
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = np.uint8(gray)
                # .. e prendo i kaypoints grazie al (simpleblob) detector
                keypoints = detector.detect(gray)
                keypoints = truncate(cv2.KeyPoint_convert(keypoints),1)
                
                #calcolo le coordinate di distanza e angolo tra i due robot
                coord = get_robot_rel_coord(trans_field,trans_field_tracked)
                
                
                
                if keypoints is not None:
                    
                    #print(f"ci sono {len(keypoints)} keypoints")
                    input_nn = [[keypoints[0][0],keypoints[0][1], comp_vals[0], comp_vals[1]]]

                    predicted_angle = model_ang.predict(input_nn)
                    predicted_dis = model_dis.predict(input_nn)

                    #print(f"l'angolo robot e il tracked (sistema del mondo) è: {coord[0]}")

                    coord[0] = map_ang_to_N(coord[0])

                    
                    '''print(f"l'angolo tra il nord e il tracked è: {coord[0]}")
                    '''
                    print(f"l'angolo predetto è {truncate(predicted_angle,1)}")
                    '''
                    print(f"la dist vera rispetto al tracked è: {coord[1]}")'''
                    print(f"l'a distanza predetta è {truncate(predicted_dis,1)}")
                    '''
                    print("-------------------------------------------------------")
                    '''
                    #DEVO ORA CONTROLLARE IN BASE ALLA DIFFERENZA TRA L'ANGOLO PREDETTO E L'ANGOLO RISPETTO AI MOTORI GOAL_COMP, CHE ALTRO NON E'
                    #CHE L'ANGOLO CHE DEVE ASSUMERE LA BUSSOLA PARTENDO DALL'ASSE POSITIVA DEI MOTORI (OPPOSTA ALLA BUSSOLA STESSA)
                    
                    # non assegno piu inutilmente il goal angle come predicted!!!!!!!!
                    # goal_angle = predicted_angle

                
                    reached_x,aligning = triangle(typeOfTriangle,predicted_angle,motor_comp,predicted_dis,reached_x,aligning)
                    #direction = "to"
                    #chase_routine(predicted_angle,motor_comp,predicted_dis,direction)
                    #avoid_obstacle_routine()
                    changing,reached_x = avoid_obstacle_routine(right_sensor,left_sensor,changing,aligning,reached_x)
                    
                    '''if keypoints is not None:
                    #PER ORA STO ASSUMENDO SOLO UN KEYPOINT!!!!
                    if len(keypoints) == 1:
                        #prendo il primo (e unico per ora) keypoint
                        keypoint = keypoints[0]
                        # mi salvo l'immagine con il nome "uguale" alla couple_pt_coord 
                        save_img(keypoint,coord)
                        
                        istant_desc = [keypoint,comp_vals,coord]
                        
                        #me lo salvo sul file txt
                        with open('your_file.txt', 'a') as f:
                            f.write("%s\n" % istant_desc)  '''

                    #TODO CI DOVRA ESSERE UN CICLO SE CI SONO PIU KEYPOINTS   
                
                else:
                    aligning = False
                    changing,reached_x = avoid_obstacle_routine(right_sensor,left_sensor,changing,aligning,reached_x)
            
            else:
                pass    
        
        else:
            changing,reached_x = avoid_obstacle_routine(right_sensor,left_sensor,changing,aligning,reached_x)


        

