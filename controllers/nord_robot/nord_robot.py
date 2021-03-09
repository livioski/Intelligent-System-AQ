from controller import Robot, Camera, DistanceSensor
import random 


def initialize_sensors_motors():

    right_sensor = DistanceSensor('right_dis_sensor')
    left_sensor = DistanceSensor('left_dis_sensor')
    DistanceSensor.enable(right_sensor,16)
    DistanceSensor.enable(left_sensor,16)

    # get the motor devices
    leftMotor = robot.getDevice('left wheel motor')
    rightMotor = robot.getDevice('right wheel motor')

    return right_sensor,left_sensor,leftMotor,rightMotor

def avoid_obstacle_routine(right_motor,left_motor,right_sensor,left_sensor,changing):
    
    if changing > 0:
        
        changing = changing - 1

    else:
        
        if right_sensor.getValue() < 999:
            
            changing = 30
            leftMotor.setVelocity(random.randint(5,15))
            rightMotor.setVelocity(0)

        elif left_sensor.getValue() < 999:
            
            changing = 30
            leftMotor.setVelocity(0)
            rightMotor.setVelocity(random.randint(5,15))

        else:
            
            leftMotor.setVelocity(10)
            rightMotor.setVelocity(10)

    return changing


if __name__ == "__main__":
    TIME_STEP = 32

    # create the Robot instance.
    robot = Robot()
   
    right_sensor, left_sensor, leftMotor,rightMotor = initialize_sensors_motors()
    
    # set the target position of the motors
    leftMotor.setPosition(float('+inf'))
    rightMotor.setPosition(float('+inf'))

    #var for avoid ob routine
    changing = 0

    while robot.step(TIME_STEP) != -1:
        
        #changing = avoid_obstacle_routine(rightMotor,leftMotor,right_sensor,left_sensor,changing)

        
        leftMotor.setVelocity(2)
        rightMotor.setVelocity(2)