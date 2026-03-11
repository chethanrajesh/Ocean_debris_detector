import numpy as np

def generate_current():

    speed = np.random.uniform(0.1,1.5)

    direction = np.random.uniform(0,2*np.pi)

    vx = speed*np.cos(direction)
    vy = speed*np.sin(direction)

    return vx,vy