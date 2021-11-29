#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fenics import *
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


mesh = RectangleMesh(Point(0,0), Point(8, 4), 35,25 )
mesh_points=mesh.coordinates()

mesh_points_x = mesh.coordinates()[:,0].T
mesh_points_y = mesh.coordinates()[:,1].T

nn = np.shape(mesh_points_x)[0]
points = np.zeros((nn,2))
for i in range (nn):
    points[i,:] = (mesh_points_x[i], mesh_points_y[i])


# In[3]:


import csv
reader = csv.reader(open("H.CSV"), delimiter=",")
x = list(reader)
Hessian =  np.array(x).astype("float") 
cov = np.linalg.inv(Hessian)

reader = csv.reader(open("kappa.CSV"), delimiter=",")
x = list(reader)
mu = np.array(x).astype("float")


# In[4]:


S = (np.diag(cov))**0.5
S = S.reshape((nn,1))
S1 = np.eye((nn))*S


# In[5]:


R1 = np.zeros((nn,nn))
for i in range (nn):
    for j in range (nn):
        R1[i,j] = cov[i,j]/(S[i,0] * S[j,0])


# In[6]:


delta = np.zeros((nn,1))
for i in range(nn):
    delta[i,0] = S[i,0]/mu[i,0]


# In[7]:


kisi = np.zeros((nn,1))
for i in range(nn):
    kisi[i,0] = (np.log( 1 + (delta[i,0])**2 ))**0.5


# In[8]:


lambdaa = np.zeros((nn,1))
for i in range(nn):
    lambdaa[i,0] = np.log(mu[i,0]) - (kisi[i,0])**2 /2


# In[9]:


rho_x = np.zeros((nn,nn)) 
for i in range (nn):
    for j in range (nn):
        rho_x[i,j] = 1/(kisi[i,0]*kisi[j,0]) * np.log( 1 + R1[i,j]*delta[i,0]*delta[j,0] )


# In[10]:


S_x = np.eye((nn))*kisi
R_x = rho_x
Z_x = S_x@R_x@S_x


# In[11]:


MAP = np.exp(lambdaa - Z_x @ np.ones((nn,1)))


# In[12]:


Hessian_x = (np.linalg.inv(Z_x))


# In[13]:


# variance of log normal dist
log_var = np.zeros((nn,nn))
for i in range(nn):
    for j in range(nn):
        log_var[i,j] = np.exp(lambdaa[i,0]+lambdaa[j,0]+0.5*(Z_x[i,i]+Z_x[j,j])) * (np.exp(Z_x[i,j])-1)


# In[14]:


def p_y (mu, Hessian,nn, y):  ## - 0.5*np.log(np.linalg.det(Hessian))
    return (nn/2)*np.log(2*pi) + 0.5*np.transpose(np.log(y)-mu) @ Hessian @ (np.log(y)-mu) + np.sum(np.log(y))


# In[15]:


def grad (mu, Hessian,nn, y):
    return np.eye((nn))*(1/y) @ (Hessian @ (np.log(y)-mu) + np.ones((nn,1)))


# In[16]:


def hess (mu, Hessian,nn, y):
    return  np.multiply( (1/y) @ np.transpose(1/y) , Hessian) - np.eye((nn)) * (np.eye((nn))*(1/y**2) @ (Hessian@(np.log(y)-mu) + np.ones((nn,1))) ) 


# In[17]:


def kinetic_MCMC (q_new,q, Hessian, invH,g):
    return .5 * np.transpose(q - q_new + invH @ g) @ Hessian @ (q - q_new + invH @ g)


# In[18]:


def kinetic_HMC (p, Hessian):
    return .5 * np.transpose(p) @ Hessian @ p 


# In[28]:


# Markov Chain Monte Carlo

dim = nn
n = 25000  # number of generated sample
q = MAP  
proposals = 0
accepted = 0

observations = np.zeros( (dim,n) )
iterations = np.zeros( (n,1) )

mu = lambdaa
Hessian = Hessian_x

rf = 1.0
dt = 0.01

for i in range(n):
    accept = False
    tally = 0
    # random p
    p = (np.random.normal(0.0,1.0,nn)) 
    p = p.reshape((nn,1)) 
    
    J = p_y (mu, Hessian,nn, q)        
    
    q_new = q + (dt * p)

    J_new = p_y (mu, Hessian,nn, q_new)
        
    a = min(1,np.exp((J - J_new )*rf))

    u1 = np.random.uniform()    
    if a>u1:
        proposals += 1
        accept = True
        q = q_new
        accepted +=1
    
    observations[:,i] = q.reshape((nn))
    iterations[i,0] = i+1
    print('iteration', i)
    print('##############################################')


# In[32]:


# MAP Hessian - Hamiltonian Monte Carlo

dim = nn
n = 250   # number of generated sample
q = MAP  
proposals = 0
accepted = 0

observations = np.zeros( (dim,n) )
iterations = np.zeros( (n,1) )

dt = 0.1

mu = lambdaa
Hessian = Hessian_x

for i in range(n):
    accept = False
    # random p
    p = (np.random.normal(0.0,1.0,nn)) 
    p = p.reshape((nn,1)) 
    
    J = p_y (mu, Hessian,nn, q)
    g = grad (mu, Hessian,nn, q)

    ######### MAP H
    M = hess (mu, Hessian,nn, MAP)
    invM = np.linalg.inv(M)
    
    
    L = np.linalg.cholesky(M)
    
    # update q 
    p1 = L @ p
    p_new = p1 - (dt/2 * g)
    q_new = q +  (dt * invM @ p_new)
    
    g_new = grad (mu, Hessian,nn, q_new)
    
    p_new = p_new - dt/2 * g_new
    
    J_new = p_y (mu, Hessian,nn, q_new)
        
    a =  min(1,np.exp((J + kinetic_HMC(p1, invM) - J_new - kinetic_HMC(p_new, invM))))
    acce = np.exp((J + kinetic_HMC(p1, invM) - J_new - kinetic_HMC(p_new, invM)))
    
    pot = J
    
    u1 = np.random.uniform()    
    if a>u1:
        proposals += 1
        accept = True
        q = q_new
        p = p_new
        pot = J_new
        accepted +=1
        
    acceptance_rate[i,0] = acce
    observations[:,i] = q.reshape((nn))
    iterations[i,0] = i+1
    print('iteration', i)
    print('##############################################')


# In[33]:


print("acceptance rate =", accepted/n) # MAP


# In[29]:


# local Hessian - Hamiltonian Monte Carlo

dim = nn
n = 25000   # number of generated sample
q = MAP  
proposals = 0
accepted = 0

observations = np.zeros( (dim,n) )
iterations = np.zeros( (n,1) )

dt = 0.3

mu = lambdaa
Hessian = Hessian_x

for i in range(n):
    accept = False
    # random p
    p = (np.random.normal(0.0,1.0,nn)) 
    p = p.reshape((nn,1)) 
    
    J = p_y (mu, Hessian,nn, q)
    g = grad (mu, Hessian,nn, q)
    
    ######### local H
    M = hess (mu, Hessian,nn, q)
    invM = np.linalg.inv(M)   
    L = np.linalg.cholesky(M)
    
    # update q 
    p1 = L @ p
    p_new = p1 - (dt/2 * g)
    q_new = q +  (dt * invM @ p_new)
    
    g_new = grad (mu, Hessian,nn, q_new)
    
    p_new = p_new - dt/2 * g_new
    
    J_new = p_y (mu, Hessian,nn, q_new)
        
    a =  min(1,np.exp((J + kinetic_HMC(p1, invM) - J_new - kinetic_HMC(p_new, invM) )))
        
    acce = np.exp((J + kinetic_HMC(p1, invM) - J_new - kinetic_HMC(p_new, invM)))
    
    pot = J
    
    u1 = np.random.uniform()    
    if a>u1:
        proposals += 1
        accept = True
        q = q_new
        p = p_new
        pot = J_new
        accepted +=1
        
    acceptance_rate[i,0] = acce
    observations[:,i] = q.reshape((nn))
    iterations[i,0] = i+1
    print('iteration', i)
    print('##############################################')


# In[61]:


# Hamiltonian Monte Carlo

dim = nn
n = 25000  # number of generated sample
q = MAP 
proposals = 0
accepted = 0

observations = np.zeros( (dim,n) )
iterations = np.zeros( (n,1) )
dt = 0.15

mu = lambdaa
Hessian = Hessian_x

for i in range(n):
    accept = False
    tally = 0
    # random p
    p = (np.random.normal(0.0,1.0,nn)) 
    p = p.reshape((nn,1)) 
    
    J = p_y (mu, Hessian,nn, q)
    g = grad (mu, Hessian,nn, q)
    
    ######### M
    M = np.eye((nn))
    
    # update q   
    p_new = p - (dt/2 * g)
    q_new = q +  (dt * p_new)
    
    g_new = grad (mu, Hessian,nn, q_new)
    
    p_new = p_new - dt/2 * g_new
    
    J_new = p_y (mu, Hessian,nn, q_new)
    
    a = min(1,np.exp((J + kinetic_HMC(p, M) - J_new - kinetic_HMC(p_new, M))))
        
    acce = np.exp((J + kinetic_HMC(p, M) - J_new - kinetic_HMC(p_new, M)))
    
    pot = J
    
    u1 = np.random.uniform()    
    if a>u1:
        proposals += 1
        accept = True
        q = q_new
        p = p_new
        pot = J_new
        accepted +=1
        
    acceptance_rate[i,0] = acce
    observations[:,i] = q.reshape((nn))
    iterations[i,0] = i+1
    print('iteration', i)
    print('##############################################')

