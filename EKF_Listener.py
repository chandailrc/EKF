#!/usr/bin/env python
#-*- coding: utf8 -*-
# from cv2 import *
from math import sqrt
import numpy as np, rospy
from threading import Thread
from matplotlib import pyplot as plt
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from time import sleep
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
from sympy import Symbol, symbols, Matrix, sin, cos, sqrt, atan
from sympy import init_printing
_FLOAT_EPS_4 = np.finfo(float).eps * 4.0
init_printing(use_latex=True)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>EKF<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<,,
def mat2euler(M, cy_thresh=None):
    M = np.asarray(M)
    if cy_thresh is None:
        try:
            cy_thresh = np.finfo(M.dtype).eps * 4
        except ValueError:
            cy_thresh = _FLOAT_EPS_4
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
    # cy: sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)
    cy = math.sqrt(r33*r33 + r23*r23)
    if cy > cy_thresh: # cos(y) not close to zero, standard form
        z = math.atan2(-r12,  r11) # atan2(cos(y)*sin(z), cos(y)*cos(z))
        y = math.atan2(r13,  cy) # atan2(sin(y), cy)
        x = math.atan2(-r23, r33) # atan2(cos(y)*sin(x), cos(x)*cos(y))
    else: # cos(y) (close to) zero, so x -> 0.0 (see above)
        # so r21 -> sin(z), r22 -> cos(z) and
        z = math.atan2(r21,  r22)
        y = math.atan2(r13,  cy) # atan2(sin(y), cy)
        x = 0.0
    return z, y, x


numstates = 61  # States
dt = 1.0/10.0  


x_pos, vx, z_pos, vz, yaw, yawR, roll, rollR, pitch, pitchR = symbols('\px, \dot\px, \pz, \dot\pz, \yaw, \dot\yaw, \wroll, \dot\wroll, \pitch, \dot\pitch')
y_pos, vy, Vxz, ax, az, ay, Axz, fxFL, fxFR, fxRL, fxRR = symbols('\py, \dot\py, \Vxz, \dot\dot\px, \dot\dot\pz, \dot\dot\py, \dot\Vxz, fxFL, fxFR, fxRL, fxRR')
fyFL, fyFR, fyRL, fyRR, fzFL, fzFR, fzRL, fzRR = symbols('fyFL, fyFR, fyRL, fyRR, fzFL, fzFR, FzRL, fzRR')

sFL, sFR, sRL, sRR, wFL, wFR, wRL, wRR, vFL, vFR, vRL, vRR,\
beta, alphaFL, alphaFR, alphaRL, alphaRR, yawA, pitchA,\
rollA, delta = symbols('sFL, sFR, sRL, sRR, wFL, wFR, wRL, wRR, vFL, vFR, vRL, vRR, beta, alphaFL, alphaFR, alphaRL, alphaRR, \dot\dot\yaw, \dot\dot\pitch, \dot\dot\wroll, delta')

zs, zsR, zsA, zs_FL, zsR_FL, zsA_FL,\
zs_FR, zsR_FR, zsA_FR, zs_RL, zsR_RL,\
zsA_RL, zs_RR, zsR_RR, zsA_RR = symbols('\zs, \dot\zs, \dot\dot\zs \zs_FL, \dot\zs_FL, \dot\dot\zs_FL,  \zs_FR, \dot\zs_FR, \dot\dot\zs_FR \zs_RL, \dot\zs_RL, \dot\dot\zs_RL \zs_RR, \dot\zs_RR, \dot\dot\zs_RR')

m, ms, h, tf, tr, d, a, b, wheelr, ks_FL, ks_FR, ks_RL, ks_RR = symbols('m, ms, h, tf, tr, d, a, b, wheelr, ks_FL, ks_FR, ks_RL, ks_RR')
c_FL, c_FR, c_RL, c_RR, m_b, m_wFL, m_wFR, m_wRL, m_wRR, kaf, g = symbols('c_FL, c_FR, c_RL, c_RR, m_b, m_wFL, m_wFR, m_wRL, m_wRR, kaf, g')
Ix, Iy, Iz, Ca_FL, Ca_FR, Ca_RL, Ca_RR, Ck_FL, Ck_FR, Ck_RL, Ck_RR = symbols('Ix, Iy, Iz, Ca_FL, Ca_FR, Ca_RL, Ca_RR, Ck_FL, Ck_FR, Ck_RL, Ck_RR')





gs = Matrix([[x_pos + vx * dt + 0.5 * ax * (dt ** 2)],
			 [vx + ax * dt],
			 [z_pos + vz * dt + 0.5 * az * (dt ** 2)],
			 [vz + az * dt],  # vz
			 [yaw + yawR * dt + 0.5 * yawA * (dt ** 2)],  # yaw
			 [yawR + yawA * dt],  # yawR
			 [roll + rollR * dt + 0.5 * rollA * (dt ** 2)],  # roll
			 [rollR + rollA * dt],  # rollR
			 [pitch + pitchR * dt + 0.5 * pitchA * (dt ** 2)],  # pitch
			 [pitchR + pitchA * dt],  # rollR
			 [y_pos + vy * dt + 0.5 * ay * (dt ** 2)],  # y_pos
			 [vy + ay * dt],  # vy
			 [Vxz],
			 [(1/m)*((fxFL + fxFR)*cos(delta) - (fyFL + fyFR)*sin(delta) + fxRL + fxRR + ms*h*yawA*roll) + vy*yaw],
			 [az],
			 [(1/m)*((fyFL + fyFR)*cos(delta) + (fxFL + fxFR)*sin(delta) + fyRL + fyRR - ms*h*rollA) - vx*yaw],
			 [Axz],
			 [-Ck_FL * sFL],                                #[fxFL],
			 [-Ck_FR * sFR],                                #[fxFR],
			 [-Ck_RL * sRL],                                #[fxRL],
			 [-Ck_RR * sRR],                                #[fxRR],
			 [-Ca_FL * alphaFL],                                #[fyFL],
			 [-Ca_FR * alphaFR],                                #[fyFR],
			 [-Ca_RL * alphaRL],                                #[fyRL],
			 [-Ca_RR * alphaRR],                                #[fyRR],
			 [ks_FL*(-zs_FL) + c_FL*(-zsR_FL) + m_wFL*g + (0.5 * m * g + ((m * ay * h) / tf)) * (b / (a + b)) - 0.5 * m * ax * (h / (a + b))],  # fzFL
			 [ks_FR*(-zs_FR) + c_FR*(-zsR_FR) + m_wFR*g + (0.5 * m * g - ((m * ay * h) / tf) * (b / (a + b))) - 0.5 * m * ax * (h / (a + b))],  # fzFR
			 [ks_RL*(-zs_RL) + c_RL*(-zsR_RL) + m_wRL*g + (0.5 * m * g + ((m * ay * h) / tf) * (b / (a + b))) + 0.5 * m * ax * (h / (a + b))],  # fzRL
			 [ks_RR*(-zs_RR) + c_RR*(-zsR_RR) + m_wRR*g + (0.5 * m * g - ((m * ay * h) / tf) * (b / (a + b))) + 0.5 * m * ax * (h / (a + b))],  # fzRR
			 [((wFL * wheelr)/vFL) - 1],  # slip ratio FL
			 [((wFR * wheelr)/vFR) - 1],  # slip ratio FR
			 [((wRL * wheelr)/vRL) - 1],  # slip ratio RL
			 [((wRR * wheelr)/vRR) - 1],  # slip ratio RR
			 [wFL],  # wheel angular velocity
			 [wFR],
			 [wRL],
			 [wRR],
			 [sqrt(vx ** 2 + vy ** 2) + yawR * ((tf / 2) - a * beta)],        # FL Actual wheel velocity
			 [sqrt(vx ** 2 + vy ** 2) + yawR * ((-tf / 2) - a * beta)],       # FR
			 [sqrt(vx ** 2 + vy ** 2) + yawR * ((tf / 2) + b * beta)],        # RL
			 [sqrt(vx ** 2 + vy ** 2) + yawR * ((-tf / 2) + b * beta)],       # RR
			 [atan((vy / vx))],     # Beta
			 [delta - atan((vy + b*yawR) / (vx + tf * yawR * 0.5))],        # alphaFL
			 [delta - atan((vy + b*yawR) / (vx - tf * yawR * 0.5))],        # alphaFR
			 [atan((-vy + b * yawR) / (vx + tr * yawR * 0.5))],             # alphaRL
			 [atan((-vy + b * yawR) / (vx - tr * yawR * 0.5))],             # alphaRR
			 [(1/Iz)*(tf*0.5*(fxFL*cos(delta)-fyFL*sin(delta)-fxFR*cos(delta)+fyFR*sin(delta)) + tr*0.5*(fxRL-fxRR) + a*(fyFL*cos(delta)+fxFL*sin(delta)+fyFR*cos(delta)+fxFR*sin(delta)) - b*(fyRL+fyRR))],        # yawA
			 [(1/Iy)*(b*(fzRR+fzRL)-a*(fzFL+fzFR))],        # pitchA
			 [ms*g*h*roll - ms*(ay + ax*rollR)*h + (fzFL + fzRR - fzFR - fzRL)*d],     # rollA
			 [delta],
			 [zs + zsR*dt + 0.5 * zsA * (dt ** 2)],  # zs sprung mass displacement]
			 [zsR + zsA * dt],  # zsR sprung mass velocity
			 [(1 / ms) * (fzFL + fzFR + fzRL + fzRR)],
			 [zs - a*pitch + tf*roll*0.5],              # zs_FL
			 [zs - a*pitch - tf*roll*0.5],              # zs_FR
			 [zs + b*pitch - tf*roll*0.5],              # zs_RL
			 [zs + b*pitch + tf*roll*0.5],              # zs_RR
			 [zsR - a*pitchR + (tf/(2*rollR))],         # zsR_FL
			 [zsR - a*pitchR - (tf/(2*rollR))],         # zsR_FR
			 [zsR + b*pitchR - (tf/(2*rollR))],         # zsR_RL
			 [zsR + b*pitchR + (tf/(2*rollR))]])        # zsR_RR
state = Matrix([x_pos, vx, z_pos, vz, yaw, yawR, roll, rollR, pitch, pitchR, y_pos, vy, Vxz, ax, az, ay, Axz, fxFL, fxFR, fxRL,
				fxRR, fyFL, fyFR, fyRL, fyRR, fzFL, fzFR, fzRL, fzRR, sFL, sFR, sRL, sRR, wFL, wFR, wRL, wRR, vFL,
				vFR, vRL, vRR, beta, alphaFL, alphaFR, alphaRL, alphaRR, yawA, pitchA, rollA, delta, zs, zsR, zsA,
				zs_FL, zs_FR, zs_RL, zs_RR, zsR_FL, zsR_FR, zsR_RL, zsR_RR])


# This formulas calculate how the state is evolving from one to the next time step



#gs


# ### Calculate the Jacobian of the Dynamic Matrix with respect to the state vector

#state

Jgs=gs.jacobian(state)

#gs[37]


'''
for i in range(0, 61):
	fh = open("Jdata1.txt", "w")
	print(Jgs[i, :], file=fh)
	fh.close()
	fh = open("Jdata1.txt", "r+")
	text_in_file = fh.read()
	fh.close()
	myfile = open("test.txt", "a")
	print(text_in_file, file=myfile)
	myfile.close()
'''


P = np.diag([1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0])
#print(P, P.shape)



# ## Process Noise Covariance Matrix Q


'''state = Matrix([x_pos, vx, z_pos, vz, yaw, yawR, roll, rollR, pitch, pitchR, y_pos, vy, Vxz, ax, az, ay, Axz, fxFL, fxFR, fxRL,
					fxRR, fyFL, fyFR, fyRL, fyRR, fzFL, fzFR, fzRL, fzRR, sFL, sFR, sRL, sRR, wFL, wFR, wRL, wRR, vFL,
					vFR, vRL, vRR, beta, alphaFL, alphaFR, alphaRL, alphaRR, yawA, pitchA, rollA, delta, zs, zsR, zsA,
					zs_FL, zs_FR, zs_RL, zs_RR, zsR_FL, zsR_FR, zsR_RL, zsR_RR])'''
Q = np.diag([0.01, 0.01, 0.01, 0.01, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
			 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
			 0.01, 0.01, 0.01, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.01, 0.01, 0.01,
			 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
#print(Q, Q.shape)


# ## Real Measurements


datafile = '/home/equaltrace/catkin_ws/src/viso2-indigo/viso2_ros/scripts/03.csv'

row1_e1, row1_e2, row1_e3, row1_e4, row2_e1, row2_e2, row2_e3, row2_e4, row3_e1, row3_e2, row3_e3, row3_e4 = np.loadtxt(datafile, delimiter=',', unpack=True)


print('Read \'%s\' successfully.' % datafile)


iterLen = len(row1_e1)
yaw_meas_S = []
pitch_meas_S = []
roll_meas_S = []
for i in range(0, iterLen):
	inMat = [[row1_e1[i], row1_e2[i], row1_e3[i]], [row2_e1[i], row2_e2[i], row2_e3[i]], [row3_e1[i], row3_e2[i], row3_e3[i]]]
	zYaw, yRoll, xPitch = mat2euler(inMat)
	yaw_meas_S.append(-zYaw)
	pitch_meas_S.append(yRoll)
	roll_meas_S.append(-xPitch)

# ## Measurement Function H


hs = Matrix([[x_pos],
			 [y_pos],
			 [yaw],
			 [roll],
			 [pitch],
			 [x_pos],
			 [y_pos],
			 [yaw]])
#hs


# In[13]:

JHs=hs.jacobian(state)
#JHs
'''
for i in range(0, 8):
	fh = open("Jdata1.txt", "w")
	print(JHs[i, :], file=fh)
	fh.close()
	fh = open("Jdata1.txt", "r+")
	text_in_file = fh.read()
	fh.close()
	myfile = open("test2.txt", "a")
	print(text_in_file, file=myfile)
	myfile.close()
'''
# ## Measurement Noise Covariance $R$


R = np.diag([1, 1, 0.01, 0.01, 0.01, 1, 1, 0.01])
#print(R, R.shape)



# ## Identity Matrix



I = np.eye(numstates)
#print(I, I.shape)


# ## Initial State



'''state = Matrix([x_pos, vx, z_pos, vz, yaw, yawR, roll, rollR, pitch, pitchR, y_pos, vy, Vxz, ax, az, ay, Axz, fxFL, fxFR, fxRL,
					fxRR, fyFL, fyFR, fyRL, fyRR, fzFL, fzFR, fzRL, fzRR, sFL, sFR, sRL, sRR, wFL, wFR, wRL, wRR, vFL,
					vFR, vRL, vRR, beta, alphaFL, alphaFR, alphaRL, alphaRR, yawA, pitchA, rollA, delta, zs, zsR, zsA,
					zs_FL, zs_FR, zs_RL, zs_RR, zsR_FL, zsR_FR, zsR_RL, zsR_RR])'''

x = np.matrix([1, 1, 1, 1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
			  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
			  1, 1, 1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1, 1, 1,
			  1, 1, 1, 1, 1, 1, 1, 1]).T
#print(x, x.shape)


# ### Put everything together as a measurement vector



x_meas_VO = row3_e4
y_meas_VO = row1_e4
yaw_meas_SEN = roll_meas_S
roll_meas_SEN = pitch_meas_S
pitch_meas_SEN = yaw_meas_S
x_meas_SEN = row3_e3
y_meas_SEN = row1_e4
yaw_meas_SEN = roll_meas_S



# In[20]:

# Preallocation for Plotting
x0 = []
x1 = []
x2 = []
x3 = []
x4 = []
x5 = []
Zx = []
Zy = []
Px = []
Py = []
Pdx= []
Pdy= []
Pddx=[]
Pddy=[]
Kx = []
Ky = []
Kdx= []
Kdy= []
Kddx=[]
dstate=[]


# # Extended Kalman Filter


m = 1
ms = 1
h = 1
tf = 1
tr = tf
d = 2*tf
a = 1
b = 1
wheelr = 1
ks_FL = 1       # Spring Constant
ks_FR = 1
ks_RL = 1
ks_RR = 1
c_FL = 1        # Damping Constant
c_FR = 1
c_RL = 1
c_RR = 1
m_b = 1         # Mass of the vehicle sans frong and rear wheels
m_wFL = 1       # Mass of wheel
m_wFR = 1
m_wRL = 1
m_wRR = 1
kaf = 1
g = 9.81
dof = 14

Ix = 1
Iy = 1
Iz = 1

Ca_FL = 1
Ca_FR = 1
Ca_RL = 1
Ca_RR = 1

Ck_FL = 1
Ck_FR = 1
Ck_RL = 1
Ck_RR = 1



ptsVisual = []
ptsOdom = []

iterI =  0

print("Processed through constants") 

def spin():
    rospy.spin()


def receivePosition(p):
    global ax, ptsVisual, f, x, P, Q, R,  iterI
    EKF([p.pose.position.x, p.pose.position.y, p.pose.position.z])
	
def EKF(p):
	global f, x, P, Q, R,  iterI
	rospy.loginfo(rospy.get_caller_id() + '  I start %s', iterI)
	measurements = np.vstack((p[2], p[0], yaw_meas_SEN[iterI], roll_meas_SEN[iterI], pitch_meas_SEN[iterI], x_meas_SEN[iterI], y_meas_SEN[iterI], yaw_meas_SEN[iterI]))
	iterI +=  1
	ptsVisual.append(p)

	
	# Time Update (Prediction)
	# ========================
	# Project the state ahead
	# see "Dynamic Matrix"

	x[0] += x[1] * dt + 0.5 * x[13] * (dt ** 2)  # x
	x[1] += x[13] * dt  # x vel
	x[2] += x[3] * dt + 0.5 * x[14] * (dt ** 2)  # z
	x[3] += x[14] * dt  # z vel
	x[4] += x[5] * dt + 0.5 * x[46] * (dt ** 2)  # yaw
	x[5] += x[46] * dt  # yaw rate
	x[6] += x[7] * dt + 0.5 * x[48] * (dt ** 2)  # roll
	x[7] += x[48] * dt  # roll rate
	x[8] += x[9] * dt + 0.5 * x[47] * (dt ** 2)  # pitch
	x[9] += x[47] * dt  # pitch rate
	x[10] += x[11] * dt + 0.5 * x[15] * (dt ** 2)  # y
	x[11] += x[15] * dt  # y vel
	x[12] = x[12]       # Vxy
	x[13] = (1/m)*((x[17]+x[18])*cos(x[49]) - (x[21]+x[22])*sin(x[49]) + x[19] + x[20] + ms*h*x[46]*x[6]) + x[11]*x[4]      #ax
	x[14] = x[14]
	x[15] = (1/m)*((x[21]+x[22])*cos(x[49]) + (x[17]+x[18])*sin(x[49]) + x[23] + x[24] - ms*h*x[48]) - x[1]*x[4]      # ay
	x[16] = x[16]       # Axy
	x[17] = -Ck_FL * x[29]       # fxFL
	x[18] = -Ck_FR * x[30]       # fxFR
	x[19] = -Ck_RL * x[31]       # fxRL
	x[20] = -Ck_RR * x[32]       # fxRR
	x[21] = -Ca_FL * x[42]       # fyFL
	x[22] = -Ca_FR * x[43]       # fyFR
	x[23] = -Ca_RL * x[44]       # fyRL
	x[24] = -Ca_RR * x[45]       # fyRR
	x[25] = ks_FL*(-x[53]) + c_FL*(-x[57]) + m_wFL*g + (0.5*m*g + ((m*h*x[15])/tf))*(b/(a+b)) - 0.5*m*x[13]*(h/(a+b))       # fzFL
	x[26] = ks_FR*(-x[54]) + c_FR*(-x[58]) + m_wFR*g + (0.5*m*g - ((m*h*x[15])/tf))*(b/(a+b)) - 0.5*m*x[13]*(h/(a+b))       # fzFR
	x[27] = ks_RL*(-x[55]) + c_RL*(-x[59]) + m_wRL*g + (0.5*m*g + ((m*h*x[15])/tr))*(a/(a+b)) + 0.5*m*x[13]*(h/(a+b))       # fzRL
	x[28] = ks_RR*(-x[60]) + c_RR*(-x[60]) + m_wRR*g + (0.5*m*g - ((m*h*x[15])/tr))*(a/(a+b)) + 0.5*m*x[13]*(h/(a+b))       # fzRR
	x[29] = x[33]*wheelr/x[37] - 1      # slip ratio FL
	x[30] = x[34]*wheelr/x[38] - 1      # slip ratio FR
	x[31] = x[35]*wheelr/x[39] - 1      # slip ratio RL
	x[32] = x[36]*wheelr/x[40] - 1      # slip ratio RR
	x[33] = x[33]       # wheel angular velocity FL
	x[34] = x[34]       # FR
	x[35] = x[35]       # RL
	x[36] = x[36]       # RR
	x[37] = sqrt(x[1]**2 + x[11]**2) + x[5]*((tf/2)-a*x[41])    # FL acutal wheel velocity
	x[38] = sqrt(x[1]**2 + x[11]**2) + x[5]*((-tf/2)-a*x[41])   # FR
	x[39] = sqrt(x[1]**2 + x[11]**2) + x[5]*((tf/2)+b*x[41])    # RL
	x[40] = sqrt(x[1]**2 + x[11]**2) + x[5]*((-tf/2)+b*x[41])   # RR
	x[41] = atan(x[11]/x[1])        # Beta
	x[42] = x[49] - atan((x[11] + b*x[5])/(x[1] + tf*x[5]*0.5)) # alpha FL
	x[43] = x[49] - atan((x[11] + b*x[5])/(x[1] - tf*x[5]*0.5)) # alpha FR
	x[44] = atan((-x[11] + b*x[5])/(x[1] + tr*x[5]*0.5))        # alpha RL
	x[45] = atan((-x[11] + b*x[5])/(x[1] - tr*x[5]*0.5))        # alpha RR
	x[46] = (1/Iz)*(tf*0.5*(x[17]*cos(x[49]) - x[21]*sin(x[49]) - x[18]*cos(x[49]) + x[22]*sin(x[49])) + tr*0.5*(x[19]-x[20]) + a*(x[21]*cos(x[49])+x[17]*sin(x[49])+x[22]*cos(x[49])+x[18]*sin(x[49])) - b*(x[23]-x[24]))      # yawA
	x[47] = (1/Iy)*(b*(x[28]+x[27]) - a*(x[26]+x[25]))  # pitchA
	x[48] = ms*g*h*x[6] - ms*(x[15] + x[13]*x[7])*h + (x[25]+x[28]-x[27]-x[26])*d       #rollA
	x[49] = x[49]       # delta
	x[50] += x[51]*dt + 0.5*x[52]*(dt**2)       # zs sprung mass displacement
	x[51] += x[52]*dt       # zsR sprung mass velocity
	x[52] = (1/ms)*(x[25] + x[26] + x[27] + x[28])       # zsA sprung mass accelration
	x[53] = x[50] - a*x[8] + tf*x[6]*0.5    # zs_FL
	x[54] = x[50] - a*x[8] - tf*x[6]*0.5    # zs_FR
	x[55] = x[50] + b*x[8] - tf*x[6]*0.5    # zs_RL
	x[56] = x[50] + b*x[8] + tf*x[6]*0.5    # zs_RR
	x[57] = x[50] - a*x[9] + tf*0.5*(1/x[7])    # zsR_FL
	x[58] = x[50] - a*x[9] - tf*0.5*(1/x[7])    # zsR_FR
	x[59] = x[50] + b*x[9] - tf*0.5*(1/x[7])    # zsR_RL
	x[60] = x[50] + b*x[9] + tf**0.5*(1/x[7])    # zsR_RR



	# Calculate the Jacobian of the Dynamic Matrix A
	# see "Calculate the Jacobian of the Dynamic Matrix with respect to the state vector"

	JA = np.matrix([[1, 0.100000000000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00500000000000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.100000000000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 1, 0.100000000000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00500000000000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.100000000000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 1, 0.100000000000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00500000000000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.100000000000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 1, 0.100000000000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00500000000000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.100000000000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0.100000000000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00500000000000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.100000000000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0.100000000000000, 0, 0, 0, 0.00500000000000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0.100000000000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, float(x[11]), 0, float(x[46]*h*ms/m), 0, 0, 0, 0, float(x[4]), 0, 0, 0, 0, 0, float(cos(x[49])/m), float(cos(x[49])/m), float(1/m), float(1/m), float(-sin(x[49])/m), float(-sin(x[49])/m), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, float(x[6]*h*ms/m), 0, 0, float((-(x[17] + x[18])*sin(x[49]) - (x[21] + x[22])*cos(x[49]))/m), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, float(-x[4]), 0, 0, float(-x[1]), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, float(sin(x[49])/m), float(sin(x[49])/m), 0, 0, float(cos(x[49])/m), float(cos(x[49])/m), 1/m, 1/m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, float(-h*ms/m), float(((x[17] + x[18])*cos(x[49]) - (x[21] + x[22])*sin(x[49]))/m), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -Ck_FL, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -Ck_FR, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -Ck_RL, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -Ck_RR, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -Ca_FL, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -Ca_FR, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -Ca_RL, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -Ca_RR, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.5*h*m/(a + b), 0, b*h*m/(tf*(a + b)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -ks_FL, 0, 0, 0, -c_FL, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.5*h*m/(a + b), 0, -b*h*m/(tf*(a + b)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -ks_FR, 0, 0, 0, -c_FR, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5*h*m/(a + b), 0, b*h*m/(tf*(a + b)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -ks_RL, 0, 0, 0, -c_RL, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5*h*m/(a + b), 0, -b*h*m/(tf*(a + b)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -ks_RR, 0, 0, 0, -c_RR],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, float(wheelr/x[37]), 0, 0, 0, float(-x[33]*wheelr/x[37]**2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, float(wheelr/x[38]), 0, 0, 0, float(-x[34]*wheelr/x[38]**2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, float(wheelr/x[39]), 0, 0, 0, float(-x[35]*wheelr/x[39]**2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, float(wheelr/x[40]), 0, 0, 0, float(-x[36]*wheelr/x[40]**2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, float(x[1]/sqrt(x[1]**2 + x[11]**2)), 0, 0, 0, float(-a*x[41] + tf/2), 0, 0, 0, 0, 0, float(x[11]/sqrt(x[1]**2 + x[11]**2)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, float(-x[5]*a), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, float(x[1]/sqrt(x[1]**2 + x[11]**2)), 0, 0, 0, float(-a*x[41] - tf/2), 0, 0, 0, 0, 0, float(x[11]/sqrt(x[1]**2 + x[11]**2)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, float(-x[5]*a), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, float(x[1]/sqrt(x[1]**2 + x[11]**2)), 0, 0, 0, float(b*x[41] + tf/2), 0, 0, 0, 0, 0, float(x[11]/sqrt(x[1]**2 + x[11]**2)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, float(x[5]*b), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, float(x[1]/sqrt(x[1]**2 + x[11]**2)), 0, 0, 0, float(b*x[41] - tf/2), 0, 0, 0, 0, 0, float(x[11]/sqrt(x[1]**2 + x[11]**2)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, float(x[5]*b), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, float(-x[11]/(x[1]**2*(1 + x[11]**2/x[1]**2))), 0, 0, 0, 0, 0, 0, 0, 0, 0, float(1/(x[1]*(1 + x[11]**2/x[1]**2))), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, float((x[11] + x[5]*b)/((1 + (x[11] + x[5]*b)**2/(x[1] + 0.5*x[5]*tf)**2)*(x[1] + 0.5*x[5]*tf)**2)), 0, 0, 0, float(-(b/(x[1] + 0.5*x[5]*tf) - 0.5*tf*(x[11] + x[5]*b)/(x[1] + 0.5*x[5]*tf)**2)/(1 + (x[11] + x[5]*b)**2/(x[1] + 0.5*x[5]*tf)**2)), 0, 0, 0, 0, 0, float(-1/((1 + (x[11] + x[5]*b)**2/(x[1] + 0.5*x[5]*tf)**2)*(x[1] + 0.5*x[5]*tf))), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, float((x[11] + x[5]*b)/((1 + (x[11] + x[5]*b)**2/(x[1] - 0.5*x[5]*tf)**2)*(x[1] - 0.5*x[5]*tf)**2)), 0, 0, 0, float(-(b/(x[1] - 0.5*x[5]*tf) + 0.5*tf*(x[11] + x[5]*b)/(x[1] - 0.5*x[5]*tf)**2)/(1 + (x[11] + x[5]*b)**2/(x[1] - 0.5*x[5]*tf)**2)), 0, 0, 0, 0, 0, float(-1/((1 + (x[11] + x[5]*b)**2/(x[1] - 0.5*x[5]*tf)**2)*(x[1] - 0.5*x[5]*tf))), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, float(-(-x[11] + x[5]*b)/((1 + (-x[11] + x[5]*b)**2/(x[1] + 0.5*x[5]*tr)**2)*(x[1] + 0.5*x[5]*tr)**2)), 0, 0, 0, float((b/(x[1] + 0.5*x[5]*tr) - 0.5*tr*(-x[11] + x[5]*b)/(x[1] + 0.5*x[5]*tr)**2)/(1 + (-x[11] + x[5]*b)**2/(x[1] + 0.5*x[5]*tr)**2)), 0, 0, 0, 0, 0, float(-1/((1 + (-x[11] + x[5]*b)**2/(x[1] + 0.5*x[5]*tr)**2)*(x[1] + 0.5*x[5]*tr))), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, float(-(-x[11] + x[5]*b)/((1 + (-x[11] + x[5]*b)**2/(x[1] - 0.5*x[5]*tr)**2)*(x[1] - 0.5*x[5]*tr)**2)), 0, 0, 0, float((b/(x[1] - 0.5*x[5]*tr) + 0.5*tr*(-x[11] + x[5]*b)/(x[1] - 0.5*x[5]*tr)**2)/(1 + (-x[11] + x[5]*b)**2/(x[1] - 0.5*x[5]*tr)**2)), 0, 0, 0, 0, 0, float(-1/((1 + (-x[11] + x[5]*b)**2/(x[1] - 0.5*x[5]*tr)**2)*(x[1] - 0.5*x[5]*tr))), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, float((a*sin(x[49]) + 0.5*tf*cos(x[49]))/Iz), float((a*sin(x[49]) - 0.5*tf*cos(x[49]))/Iz), float(0.5*tr/Iz), float(-0.5*tr/Iz), float((a*cos(x[49]) - 0.5*tf*sin(x[49]))/Iz), float((a*cos(x[49]) + 0.5*tf*sin(x[49]))/Iz), float(-b/Iz), float(-b/Iz), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, float((a*(x[17]*cos(x[49]) + x[18]*cos(x[49]) - x[21]*sin(x[49]) - x[22]*sin(x[49])) + 0.5*tf*(-x[17]*sin(x[49]) + x[18]*sin(x[49]) - x[21]*cos(x[49]) + x[22]*cos(x[49])))/Iz), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -a/Iy, -a/Iy, b/Iy, b/Iy, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, g*h*ms, float(-x[13]*h*ms), 0, 0, 0, 0, 0, float(-x[7]*h*ms), 0, -h*ms, 0, 0, 0, 0, 0, 0, 0, 0, 0, d, -d, -d, d, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0.100000000000000, 0.00500000000000000, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0.100000000000000, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1/ms, 1/ms, 1/ms, 1/ms, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0.5*tf, 0, -a, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, -0.5*tf, 0, -a, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, -0.5*tf, 0, b, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0.5*tf, 0, b, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, float(-tf/(2*x[7]**2)), 0, -a, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, float(tf/(2*x[7]**2)), 0, -a, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, float(tf/(2*x[7]**2)), 0, b, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, float(-tf/(2*x[7]**2)), 0, b, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
					])

	# Project the error covariance ahead
	P = JA*P*JA.T + Q

	# Measurement Update (Correction)
	# ===============================
	# Measurement Function
	hx = np.matrix([[float(x[0])],
					[float(x[10])],
					[float(x[4])],
					[float(x[6])],
					[float(x[8])],
					[float(x[0])],
					[float(x[10])],
					[float(x[4])],
					])


	JH = np.matrix([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])


	S = JH*P*JH.T + R
	K = (P*JH.T) * np.linalg.inv(S)

	# Update the estimate via
	Z = measurements[:, 0].reshape(JH.shape[0], 1)
	y = Z - (hx)                         # Innovation or Residual
	x = x + (K*y)

	# Update the error covariance
	P = (I - (K*JH))*P


	# Save states for Plotting
	x0.append(float(x[0]))
	x1.append(float(x[1]))
	x2.append(float(x[2]))
	x3.append(float(x[3]))
	x4.append(float(x[4]))
	Zx.append(float(Z[0]))
	Zy.append(float(Z[1]))
	Px.append(float(P[0,0]))
	Py.append(float(P[1,1]))
	Pdx.append(float(P[2,2]))
	Pdy.append(float(P[3,3]))
	Pddx.append(float(P[4,4]))
	Kx.append(float(K[0,0]))
	Ky.append(float(K[1,0]))
	Kdx.append(float(K[2,0]))
	Kdy.append(float(K[3,0]))
	Kddx.append(float(K[4,0]))

	rospy.loginfo(rospy.get_caller_id() + '  I finished %s', iterI-1)
	return 


def receiveOdom(p):
    global ax, ptsOdom, f
    ptsOdom.append([p.pose.pose.position.x, p.pose.pose.position.y, p.pose.pose.position.z])


if __name__ == "__main__":
    rospy.init_node("dissertacao")
    Thread(target = spin).start()
rospy.Subscriber("/mono_odometer/pose", PoseStamped, receivePosition)
rospy.Subscriber("/odom", Odometry, receiveOdom)
