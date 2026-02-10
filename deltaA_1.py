"""
Created on Thu Jan 28 11:08:30 2021

@author: Andrés
"""

#######################
#### M O D U L O S ####
#######################

import numpy as np
import math 
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

#######################

#######################
## F U N C I O N E S ##
#######################
a = 1.0
h = 1
k1 = 0.1
k2 = 0.3
zz = np.linspace(0.0,6.0,1000)


def d(z):
    w = (36.0*a*pow(h,2.0)*k1*pow(k2,2.0))/(pow(1.0+z, 3.0*math.sqrt(a)*k2)-24.0*math.sqrt(a)*pow(h,2.0)*k1*k2)
    return w

vect1=[]

for i in range(len(zz)):
    aa = d(zz[i])
    vect1.append(aa)

def d1(p, z):
    x = (p[1]*(1.0 - p[0]))/(1.0 - p[0] + p[0]*(pow((1.0 + z), p[1])))
    return x

# p[0] es Xi 
# p[1] es n
#############################
##### MINIMOS CUADRADOS #####
#############################

param_list = []

def residuos(p, z, vect1):
    y_modelo = d1(p, zz)
    plt.clf()
    plt.plot(zz,vect1,'o',zz,y_modelo,'r-')
    plt.pause(0.05)
    param_list.append(p)
    return y_modelo - vect1

parametros_iniciales=[1.0, 1.0]  # Ajusta
res = least_squares(residuos, parametros_iniciales, args=(zz, vect1),  verbose=1)

# Estos son los parámetros hallados:
print('parámetros hallados')
print(res.x)


#Calculamos la matriz de covarianza "pcov"
def calcular_cov(res,vect1):
    U, S, V = np.linalg.svd(res.jac, full_matrices=False)
    threshold = np.finfo(float).eps * max(res.jac.shape) * S[0]
    S = S[S > threshold]
    V = V[:S.size]
    pcov = np.dot(V.T / S**2, V)#

    s_sq = 2 * res.cost / (len(vect1) - len(res.x))
    pcov = pcov * s_sq
    return pcov

pcov = calcular_cov(res,vect1)

# De la matriz de covarinza podemos obtener los valores de desviación estándar
# de los parametros hallados
pstd = np.sqrt(np.diag(pcov))

print('Parámetros hallados (con incertezas):')
for i,param in enumerate(res.x):
    print('parametro[{:d}]: {:5.3f} ± {:5.3f}'.format(i,param,pstd[i]/2))

y_modelo = d1(res.x, zz)

np.savetxt('deltaA1.txt', y_modelo)

#######################
### G R A F I C A S ###
#######################    

figDeltaT1 = plt.figure()
plt.plot(zz, vect1,  'o', markersize=4, label='Model A')
plt.plot(zz, y_modelo, 'r-', label='$\delta_I$')
plt.xlabel("z")
plt.ylabel("$\delta(z)$")
plt.legend(loc='best')
#plt.tight_layout()
figDeltaT1.savefig('deltaA_1.pdf')
plt.show()

#vect3=[] # Este vector es la función d1 evaluada con los resultados del diff evol #

#for i in range(len(zz)):
#    aa = d1(zz[i], result.x[0], result.x[1])
#    vect3.append(aa)

#figCompHz1s = plt.figure()
#plt.plot(zz, vect1, 'b-', linewidth = 1, label = '$\delta(z)$')
#plt.plot(zz, vect3, 'r-', linewidth = 1, label = '$\delta_1(z, \Xi)$ 2.37')
#plt.legend(loc = 1, numpoints = 1, fontsize = 9)
#plt.xlabel("z", fontsize = 11)
#plt.ylabel("$\delta(z)$", fontsize = 11)
#figCompHz1s.savefig('deltazeta.pdf')
#plt.show()


