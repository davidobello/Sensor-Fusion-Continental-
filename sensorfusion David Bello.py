import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#David Bello's code

#constants used to create to the obstacle
S_O_S = 343 #speed of sounds in m/s
Grid_Size = 100 #Total size of our obstacle grid
Cell_Size = 0.1 #size of each cell in obstacle grid

#sensor angles relative to robot orientation in radians
Sensor_Angles = np.array([-45,45,-135, 135])*(math.pi/180)

#importing the artificial data from our CSV file into pandas DataFrame
def load_data(robot):
    df = pd.read_csv(robot)
    df.iloc[:,4:] *=S_O_S/2 #Extracting obstacle distance for all 4 sensors
    return df

#calculating the global coordinates from obstacle positions
def obtain_obstacle_posit(df):
    Obstacle_Positions = []

#Assigning robot orientation from our CSV file
    for _, row in df.iterrows():
        x_robot,y_robot,theta0 = row[1], row[2], row[3]
        distances = row[4:].values   #we extract the individual distances of the ultrasound from our csv file

        for i in range(4):
            angle = theta0 + Sensor_Angles[i]
            Xo = x_robot + distances[i]*math.cos(angle)
            Yo = y_robot + distances[i]*math.sin(angle)
            Obstacle_Positions.append((Xo, Yo))

    return np.array(Obstacle_Positions)

#Obtaining obstacle grid from obstacle positions
def gen_obstacle_grid(obstacles):
    grid =np.zeros((Grid_Size,Grid_Size))

    for (x,y) in obstacles:
        Grid_X = int((x/Cell_Size + Grid_Size/2))
        Grid_Y = int((y/Cell_Size + Grid_Size/2))

        if 0 <= Grid_X <Grid_Size and 0 <= Grid_Y <Grid_Size:
            grid[Grid_X, Grid_Y] = 1    #occupied grid is taken as 1

    return grid

#Visualize our grid using imported matplotlib.pyplot
def Visualize_Grid(grid):
    plt.imshow(grid, cmap="gray_r", origin="lower")   #we represent it in grayscale i.e black and white
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Obstacle Grid Map")    #Title of our visualization
    plt.show()

#Executing the Program
df = load_data("robot.csv")
obstacles = obtain_obstacle_posit(df)
obstacle_grid = gen_obstacle_grid(obstacles)
Visualize_Grid(obstacle_grid)
