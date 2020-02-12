#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[18]:


data = np.loadtxt("notas_andes.dat", skiprows=1)
Y = data[:,4]
X = data[:,:4]


# In[3]:


#plt.figure(figsize=(8,8))
#plt.plot(mu,y)
#plt.scatter(mu,y)
#plt.xlabel("mu")
#plt.suptitle(valor, fontsize=20)


# In[4]:


sigma=0.1


# In[36]:


w=2*np.ones(len(X))/sigma**2


# In[44]:


beta=np.ones(len(X[0]+1))
c=beta[0]
i


# In[48]:


def loglikelihood(x, y, beta,sigma):
    y_model = y_test(x, beta)
    sum=1/(np.power(2*np.pi,0.5)*sigma)
    for i in range(len(x)):
        sum+=(y_model[i]-y)**2
    return sum


# In[40]:


def metropolis(x_obs, y_obs, N = len(X)):
    l_param = [np.array([1/8000.0, 10.0])]
    sigma_param = np.array([1E-4, 0.5])
    n_param = len(sigma_param)
    logposterior = [0]
    for i in range(1,N):
        propuesta  = l_param[i-1] + np.random.normal(size=n_param)*sigma_param
        #print(propuesta)
        logposterior_viejo = loglikelihood(x_obs, y_obs, l_param[i-1]) + logprior(l_param[i-1])
        logposterior_nuevo = loglikelihood(x_obs, y_obs, propuesta) + logprior(propuesta)

        r = min(1,np.exp(logposterior_nuevo-logposterior_viejo))
        alpha = np.random.random()
        if(alpha<r):
            l_param.append(propuesta)
            logposterior.append(logposterior_nuevo)
        else:
            l_param.append(l_param[i-1])
            logposterior.append(logposterior_viejo)
    
    l_param = np.array(l_param)
    l_param = l_param[N//10:,:] # descartamos el primer 10% de la cadena
    logposterior = np.array(logposterior)
    logposterior = logposterior[N//10:]
    return l_param, logposterior


# In[41]:


metropolis(X[0],Y)


# In[38]:





# In[34]:


def prob(m,c,sigma,x,y):
    sum=0
    for i in range(len(w)):
        sum+=(m[0]*x[i,0]+m[1]*x[i,1]+m[2]*x[i,2]+m[3]*x[i,3]+c-y[i])**2
    return sigma**-len(x)*np.exp(-sum/(2*sigma**2))


# # Valores óptimos

# In[32]:


def mo(w,x,y):
    return (beta(w,x)*p(w,x,y)-gama(w,x)*q(w,y))/(alfa(w,x)*beta(w,x)-gama(w,x)**2)
def co(w,x,y):
    return (alfa(w,x)*q(w,y)-gama(w,x)*p(w,x,y))/(alfa(w,x)*beta(w,x)-gama(w,x)**2)


# In[31]:


def alfa(w,x):
    sum=0
    for i in range(len(w)):
        sum+=w[i]*x[i]**2
    return sum
def beta(w,x):
    sum=0
    for i in range(len(w)):
        sum+=w[i]
    return sum
def gama(w,x):
    sum=0
    for i in range(len(w)):
        sum+=w[i]*x[i]
    return sum
def p(w,x,y):
    sum=0
    for i in range(len(w)):
        sum+=w[i]*x[i]*y[i]
    return sum
def q(w,y):
    sum=0
    for i in range(len(w)):
        sum+=w[i]*y[i]
    return sum


# In[33]:


mo(w,X[:,0],Y)


# In[ ]:




