#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'widget')
import time
import itk
from itkwidgets import view
import pickle
import copy


# In[2]:


def index_matrix_generation():
    index_matrix=np.zeros((720,720,768))
    return index_matrix
index_matrix=index_matrix_generation()


# In[3]:


def seed_generation():
    #set starting seed at trechea
    mu = 6 #
    sigma = 2 
    rand_pos = np.random.normal(mu, sigma, 3)
    seed=np.array([int(rand_pos[0]),int(rand_pos[1])-20,80])
    return seed
seed=seed_generation()


# ## Main branch parameter generation

# In[4]:


def parameter_generation(cl_mean,cl_sig,r_mean,r_sig,phi_min,phi_max,theta_min,theta_max,wall_mean,wall_sig):
    cl=np.random.normal(cl_mean,cl_sig)
    if cl<0:
        cl=0
    r=np.random.normal(r_mean,r_sig)
    phi=np.random.randint(phi_min,phi_max)/180*np.pi
    theta=np.random.randint(theta_min,theta_max)/180*np.pi
    wall=np.random.normal(wall_mean,wall_sig)
    return np.array([cl,r,phi,theta,wall])


# In[5]:


def normalized_parameter_generation(cl_mean,cl_sig,rn_mean,rn_sig,phi_min,phi_max,theta_min,theta_max,wall_mean,wall_sig,Trachea_r,limit=[0,0]):
    cl=np.random.normal(cl_mean,cl_sig)
    if cl<0:
        cl=0
#     if cl>=1.5*limit[0]:
#         cl=1.5*limit[0]
    r=np.random.normal(rn_mean,rn_sig)*Trachea_r
    phi=np.random.randint(phi_min,phi_max)/180*np.pi
    theta=np.random.randint(theta_min,theta_max)/180*np.pi
    if limit[1] >0:
        r=min(limit[1],r)
    wall=np.random.normal(wall_mean,wall_sig)
    return np.array([cl,r,phi,theta,wall])


# In[6]:


trachea=parameter_generation(109,14,23.9,2.9,85,95,18,20,2.6,0.24)## radius needed and centerline length needed
R=trachea[1]
rm=normalized_parameter_generation(25.5,2.5,0.84,0.13,175,195,30,35,2.3,0.33,R)
ru=normalized_parameter_generation(15,4,0.59,0.17,180,210,85,105,1.8,0.3,R,rm)
br=normalized_parameter_generation(29,3,0.653,0.11,175,195,30,45,1.82,0.25,R,rm)
r45=normalized_parameter_generation(19,4,0.38,0.09,180,225,60,90,1.6,0.15,R,br)
r6=normalized_parameter_generation(10,4,0.36,0.1,105,145,60,90,1.61,0.3,R,br)
rl7=normalized_parameter_generation(10.5,1.25,0.463,0.08,175,195,20,40,1.74,0.22,R,br)
r7=normalized_parameter_generation(16,5,0.27,0.06,-105,-30,30,45,1.45,0.17,R,rl7)
rl=normalized_parameter_generation(10,4,0.41,0.1,175,195,20,40,1.6,0.21,R,rl7)
r8=normalized_parameter_generation(12.5,1.5,0.296,0.073,195,255,45,60,1.51,0.17,R)
r910=normalized_parameter_generation(11,3.5,0.366,0.083,175,195,20,40,1.61,0.15,R,rl)
r9=normalized_parameter_generation(15,3.5,0.262,0.06,175,195,20,45,1.38,0.21,R,r910)
r10=normalized_parameter_generation(13,3.5,3.03,0.074,-30,30,0,45,1.45,0.16,R,r910)
r1=normalized_parameter_generation(13,3.5,0.33,0.06,-90,180,135,180,1.5,0.22,R,ru)
r2=normalized_parameter_generation(11,3,0.31,0.061,90,150,75,100,1.48,0.16,R,ru)
r3=normalized_parameter_generation(12,3.75,0.35,0.096,225,300,80,120,1.61,0.18,R,ru)
r4=normalized_parameter_generation(12.5,4,0.27,0.07,150,180,60,90,1.4,0.16,R,r45)
r5=normalized_parameter_generation(14,4,0.29,0.06,195,225,60,90,1.47,0.17,R,r45)

lm=normalized_parameter_generation(50,5,0.7,0.11,-15,5,30,35,2.06,0.34,R)
lu=normalized_parameter_generation(13,3,0.58,0.18,-45,0,85,105,1.92,0.3,R,lm)
ll6=normalized_parameter_generation(12.5,2.5,0.6,0.13,-15,5,30,45,1.97,0.32,R,lm)
l6=normalized_parameter_generation(10,2.8,0.41,0.15,55,105,50,90,1.71,0.22,R,ll6)
ll=normalized_parameter_generation(17,4,0.47,0.1,-15,5,30,45,1.76,0.15,R,ll6)
l8=normalized_parameter_generation(12.5,4,0.32,0.07,-75,5,45,60,1.57,0.15,R,ll)
l910=normalized_parameter_generation(10,3,0.42,0.1,-25,5,15,45,1.66,0.15,R,ll)
l9=normalized_parameter_generation(15,4.75,0.3,0.075,-35,0,20,45,1.48,0.21,R,l910)#cl=15
l10=normalized_parameter_generation(15,4.75,0.32,0.075,5,25,20,45,1.5,0.17,R,l910)#cl=15
l123=normalized_parameter_generation(11.5,3,0.454,0.125,-90,90,160,180,1.75,0.19,R,lu)
l12=normalized_parameter_generation(6,2.5,0.33,0.1,120,240,120,180,1.56,0.37,R,l123)
l1=normalized_parameter_generation(13,3.5,0.263,0.08,90,150,150,180,13.8,0.188,R,l12)
l2=normalized_parameter_generation(12.5,3,0.225,0.06,150,210,150,180,1.27,0.24,R,l12)
l3=normalized_parameter_generation(10,3,0.33,0.09,-60,60,120,150,1.57,0.175,R,l123)
l45=normalized_parameter_generation(14,5,0.40,0.1,-15,30,60,75,1.62,0.41,R,lu,)
l4=normalized_parameter_generation(12.5,5,0.264,0.05,-15,30,60,75,1.36,0.16,R,l45)
l5=normalized_parameter_generation(12.5,3.5,0.26,0.06,15,30,45,60,1.35,0.16,R,l45)


# ## One-D Tree generation

# In[7]:


def main_branch_generation(seed,index_matrix,branch,base=[0,0,0,0],label=60,vanish=1,bifurcate=1):
    length=int(branch[0]*2)
    phi=branch[2]
    theta=branch[3]
    base_r=int(base[1])
    x=np.zeros((length))
    y=np.zeros((length))
    z=np.zeros((length))
    x1=x
    y1=y
    z1=z
    line=np.zeros((length))
    line1=line
    prob_v=np.random.binomial(1,vanish,1)[0]
    prob_b=np.random.binomial(1,bifurcate,1)[0]
    for i in range (0,length):
        x[i]=int((base_r+i)*np.sin(theta)*np.cos(phi))+seed[0]
        y[i]=int((base_r+i)*np.sin(theta)*np.sin(phi))+seed[1]
        z[i]=int((base_r+i)*np.cos(theta))+seed[2]
        line[i]=label
        index_matrix[int(x[i])+360,int(y[i])+360,int(z[i])]=line[i]
        if prob_v==0:
            index_matrix[int(x[i])+360,int(y[i])+360,int(z[i])]=0
    end=length-1
#     end=np.array([x[end],y[end],z[end]])
#     if prob_v==0:
#         end=seed
#     if prob_b==0:
#         theta1=theta+np.random.randint(15,30)/180*np.pi
#         for j in range (0,length):
#             x1[j]=int((base_r+j)*np.sin(theta1)*np.cos(phi))+seed[0]
#             y1[j]=int((base_r+j)*np.sin(theta1)*np.sin(phi))+seed[1]
#             z1[j]=int((base_r+j)*np.cos(theta1))+seed[2]
#             line1[j]=label
#             index_matrix[int(x1[j])+360,int(y1[j])+360,int(z1[j])]=line1[j]
#         x=np.hstack((x,x1))
#         y=np.hstack((y,y1))
#         z=np.hstack((z,z1))
    info=np.array([(x[0],y[0],z[0]),(x[end],y[end],z[end])])
    return index_matrix,info


# In[8]:


def s_branch_generation (index_matrix,f_branch,seed,label=100):
    length_factor=np.random.normal(7,1,2)/10
    radius_factor=np.random.normal(6.5,0.8,2)/10
#     radius_factor=np.array([0.5,0.5])
    theta_1=f_branch[3]+np.random.randint(5,22)/180*np.pi
    theta_2=f_branch[3]-np.random.randint(5,22)/180*np.pi
    phi_1=np.random.randint(15,20)/180+f_branch[2]
    phi_2=f_branch[2]-np.random.randint(10,20)/180
    length_1=int(length_factor[0]*f_branch[0]*2)
    length_2=int(length_factor[1]*f_branch[0]*2)

    x=np.zeros((length_1))
    y=np.zeros((length_1))
    z=np.zeros((length_1))
    x1=np.zeros((length_2))
    y1=np.zeros((length_2))
    z1=np.zeros((length_2))
    for i in range (0,length_1):
        x[i]=int((f_branch[1]+i)*np.sin(theta_1)*np.cos(phi_1))+seed[1,0]
        y[i]=int((f_branch[1]+i)*np.sin(theta_1)*np.sin(phi_1))+seed[1,1]
        z[i]=int((f_branch[1]+i)*np.cos(theta_1))+seed[1,2]
        
        index_matrix[int(x[i])+360,int(y[i])+360,int(z[i])]=label

    l=len(x)-1
    s_branch=[(x[0],y[0],z[0]),(x[l],y[l],z[l])]
    


    for j in range (0,length_2):
        x1[j]=int((f_branch[1]+j)*np.sin(theta_2)*np.cos(phi_2))+seed[1,0]
        y1[j]=int((f_branch[1]+j)*np.sin(theta_2)*np.sin(phi_2))+seed[1,1]
        z1[j]=int((f_branch[1]+j)*np.cos(theta_2))+seed[1,2]
        
        index_matrix[int(x1[j])+360,int(y1[j])+360,int(z1[j])]=label
#         index_matrix1[int(x1[j])+360,int(y1[j])+360,int(z1[j])]=label
    l1=length_2-1
    s1_branch=[(x1[0],y1[0],z1[0]),(x1[l1],y1[l1],z1[l1])]
#     x=np.hstack((x,x1))
#     y=np.hstack((y,y1))
#     z=np.hstack((z,z1))
    radius=radius_factor*f_branch[1]
#     print (s1_branch)

    ends=np.array([(length_1,radius[0],phi_1,theta_1,x[0],y[0],z[0],x[l],y[l],z[l]),(length_2,radius[1],phi_2,theta_2,x1[0],y1[0],z1[0],x1[l1],y1[l1],z1[l1])])
    return index_matrix,ends


# In[9]:


def ss_branch_generation0 (index_matrix,f_branch,label=60):
    length_factor=np.random.normal(7.5,0.3,2)/10
    radius_factor=np.random.normal(6.5,0.8,2)/10
    
#     radius_factor=np.array([0.5,0.5])
    theta_1=f_branch[3]+np.random.randint(5,22)/180*np.pi
    theta_2=f_branch[3]-np.random.randint(5,22)/180*np.pi
    phi_1=np.random.randint(15,20)/180+f_branch[2]
    phi_2=f_branch[2]-np.random.randint(10,20)/180
    length_1=int(length_factor[0]*f_branch[0]*2)
    if length_1<1:
        length_1=2
    length_2=int(length_factor[1]*f_branch[0]*2)
    if length_2<1:
        length_2=2
#     print (length_1,length_2)
    x=np.zeros((length_1))
    y=np.zeros((length_1))
    z=np.zeros((length_1))
    x1=np.zeros((length_2))
    y1=np.zeros((length_2))
    z1=np.zeros((length_2))
    for i in range (0,length_1):
        x[i]=int((f_branch[1]+i)*np.sin(theta_1)*np.cos(phi_1))+f_branch[7]
        y[i]=int((f_branch[1]+i)*np.sin(theta_1)*np.sin(phi_1))+f_branch[8]
        z[i]=int((f_branch[1]+i)*np.cos(theta_1))+f_branch[9]
        
        index_matrix[int(x[i])+360,int(y[i])+360,int(z[i])]=label

    l=length_1-1
    s_branch=[(x[0],y[0],z[0]),(x[l],y[l],z[l])]
    


    for j in range (0,length_2):
        x1[j]=int((f_branch[1]+j)*np.sin(theta_2)*np.cos(phi_2))+f_branch[7]
        y1[j]=int((f_branch[1]+j)*np.sin(theta_2)*np.sin(phi_2))+f_branch[8]
        z1[j]=int((f_branch[1]+j)*np.cos(theta_2))+f_branch[9]
        
        index_matrix[int(x1[j])+360,int(y1[j])+360,int(z1[j])]=label

    l1=length_2-1
    s1_branch=[(x1[0],y1[0],z1[0]),(x1[l1],y1[l1],z1[l1])]
#     x=np.hstack((x,x1))
#     y=np.hstack((y,y1))
#     z=np.hstack((z,z1))
    radius=radius_factor*f_branch[1]
#     print (s1_branch)
#     print (radius)
    ends=np.array([(length_1,radius[0],phi_1,theta_1,x[0],y[0],z[0],x[l],y[l],z[l]),(length_2,radius[1],phi_2,theta_2,x1[0],y1[0],z1[0],x1[l1],y1[l1],z1[l1])])
    return index_matrix,ends

def ss_branch_generation(index_matrix,ends):
    f_branch_0=ends[0]
#     print(f_branch_0)
    f_branch_1=ends[1]
#     print(f_branch_1)
    index_matrix,ends1=ss_branch_generation0(index_matrix,f_branch_0)
    index_matrix,ends2=ss_branch_generation0(index_matrix,f_branch_1)
    ends=np.vstack((ends1,ends2))

    return index_matrix,ends
    


# In[10]:


#tree generation

index_matrix,trachea_i=main_branch_generation(seed,index_matrix,trachea)
#rmb
index_matrix,rm_i=main_branch_generation(trachea_i[1],index_matrix,rm,trachea)
#rul
index_matrix,ru_i=main_branch_generation(rm_i[1],index_matrix,ru,rm)
#bronlnt
index_matrix,br_i=main_branch_generation(rm_i[1],index_matrix,br,rm)
#rb4+5
index_matrix,r45_i=main_branch_generation(br_i[1],index_matrix,r45,br)
#rb6
index_matrix,r6_i=main_branch_generation(br_i[1],index_matrix,r6,br)
#rll7
index_matrix,rl7_i=main_branch_generation(br_i[1],index_matrix,rl7,br)
#rb7(future needs a vanish factor)
index_matrix,r7_i=main_branch_generation(rl7_i[1],index_matrix,r7,rl7)
#rll
index_matrix,rl_i=main_branch_generation(rl7_i[1],index_matrix,rl7)
#rb8(needs bifurcation factor)
index_matrix,r8_i=main_branch_generation(rl_i[1],index_matrix,r8,rl)
#rb9+10(needs a vanish factor)
index_matrix,r910_i=main_branch_generation(rl_i[1],index_matrix,r910,rl)
#rb9
index_matrix,r9_i=main_branch_generation(r910_i[1],index_matrix,r9,r910)
#rb10
index_matrix,r10_i=main_branch_generation(r910_i[1],index_matrix,r10,r910)
#rb1
index_matrix,r1_i=main_branch_generation(ru_i[1],index_matrix,r1,ru)
#rb2
index_matrix,r2_i=main_branch_generation(ru_i[1],index_matrix,r2,ru)
#rb3
index_matrix,r3_i=main_branch_generation(ru_i[1],index_matrix,r3,ru)
#rb4
index_matrix,r4_i=main_branch_generation(r45_i[1],index_matrix,r4,r45)
#rb5
index_matrix,r5_i=main_branch_generation(r45_i[1],index_matrix,r5,r45)
#lmb
index_matrix,lm_i=main_branch_generation(trachea_i[1],index_matrix,lm,trachea)
#lul
index_matrix,lu_i=main_branch_generation(lm_i[1],index_matrix,lu,lm)
#llb6
index_matrix,ll6_i=main_branch_generation(lm_i[1],index_matrix,ll6,lm)
#lb6
index_matrix,l6_i=main_branch_generation(ll6_i[1],index_matrix,l6,ll6)
#llb
index_matrix,ll_i=main_branch_generation(ll6_i[1],index_matrix,ll,ll6)
#lb8
index_matrix,l8_i=main_branch_generation(ll_i[1],index_matrix,l8,ll)
#lb9+10
index_matrix,l910_i=main_branch_generation(ll_i[1],index_matrix,l910,ll)
#lb9
index_matrix,l9_i=main_branch_generation(l910_i[1],index_matrix,l9,l910)
#lb10
index_matrix,l10_i=main_branch_generation(l910_i[1],index_matrix,l10,l910)
#lb1+2+3
index_matrix,l123_i=main_branch_generation(lu_i[1],index_matrix,l123,lu)
#lb1+2
index_matrix,l12_i=main_branch_generation(l123_i[1],index_matrix,l12,l123)
#lb3
index_matrix,l3_i=main_branch_generation(l123_i[1],index_matrix,l3,l123)
#lb1
index_matrix,l1_i=main_branch_generation(l12_i[1],index_matrix,l1,l12)
#lb2
index_matrix,l2_i=main_branch_generation(l12_i[1],index_matrix,l2,l12)
#lb4+5
index_matrix,l45_i=main_branch_generation(lu_i[1],index_matrix,l45,lu)
#lb4
index_matrix,l4_i=main_branch_generation(l45_i[1],index_matrix,l4,l45)
#lb5
index_matrix,l5_i=main_branch_generation(l45_i[1],index_matrix,l5,l45)    


# In[11]:


index_matrix,sr1=s_branch_generation(index_matrix,r1,r1_i)
index_matrix,sr2=s_branch_generation(index_matrix,r2,r2_i)
index_matrix,sr3=s_branch_generation(index_matrix,r3,r3_i)
index_matrix,sr4=s_branch_generation(index_matrix,r4,r4_i)
index_matrix,sr5=s_branch_generation(index_matrix,r5,r5_i)
index_matrix,sr6=s_branch_generation(index_matrix,r6,r6_i)
index_matrix,sr7=s_branch_generation(index_matrix,r7,r7_i)
index_matrix,sr8=s_branch_generation(index_matrix,r8,r8_i)
index_matrix,sr9=s_branch_generation(index_matrix,r9,r9_i)
index_matrix,sr10=s_branch_generation(index_matrix,r10,r10_i)
index_matrix,sl1=s_branch_generation(index_matrix,l1,l1_i)
index_matrix,sl2=s_branch_generation(index_matrix,l2,l2_i)
index_matrix,sl3=s_branch_generation(index_matrix,l3,l3_i)
index_matrix,sl4=s_branch_generation(index_matrix,l4,l4_i)
index_matrix,sl5=s_branch_generation(index_matrix,l5,l5_i)
index_matrix,sl6=s_branch_generation(index_matrix,l6,l6_i)
index_matrix,sl8=s_branch_generation(index_matrix,l8,l8_i)
index_matrix,sl9=s_branch_generation(index_matrix,l9,l9_i)
index_matrix,sl10=s_branch_generation(index_matrix,l10,l10_i)


# In[12]:


index_matrix,ssr1=ss_branch_generation(index_matrix,sr1)
index_matrix,ssr2=ss_branch_generation(index_matrix,sr2)
index_matrix,ssr3=ss_branch_generation(index_matrix,sr3)
index_matrix,ssr4=ss_branch_generation(index_matrix,sr4)
index_matrix,ssr5=ss_branch_generation(index_matrix,sr5)
index_matrix,ssr6=ss_branch_generation(index_matrix,sr6)
index_matrix,ssr7=ss_branch_generation(index_matrix,sr7)
index_matrix,ssr8=ss_branch_generation(index_matrix,sr8)
index_matrix,ssr9=ss_branch_generation(index_matrix,sr9)
index_matrix,ssr10=ss_branch_generation(index_matrix,sr10)
index_matrix,ssl1=ss_branch_generation(index_matrix,sl1)
index_matrix,ssl2=ss_branch_generation(index_matrix,sl2)
index_matrix,ssl3=ss_branch_generation(index_matrix,sl3)
index_matrix,ssl4=ss_branch_generation(index_matrix,sl4)
index_matrix,ssl5=ss_branch_generation(index_matrix,sl5)
index_matrix,ssl6=ss_branch_generation(index_matrix,sl6)
index_matrix,ssl8=ss_branch_generation(index_matrix,sl8)
index_matrix,ssl9=ss_branch_generation(index_matrix,sl9)
index_matrix,ssl10=ss_branch_generation(index_matrix,sl10)


# In[13]:


index_matrix1=np.copy(index_matrix)


# ## Airway Modeling

# In[14]:


def airway_modeling(index_matrix,info,branch,num_id=60):
#     start=time.time()
    P=info[0]+[360,360,0]
    Q=info[1]+[360,360,0]
    AB = Q - P;
    temp = AB / np.dot(AB,AB);
    r=branch[1]
    ri=branch[1]-2*branch[4]
    lmin=int(min(P[0],Q[0])-1.2*int(r))
    lmax=int(max(P[0],Q[0])+1.2*int(r))
    wmin=int(min(P[1],Q[1])-1.2*int(r))
    wmax=int(max(P[1],Q[1])+1.2*int(r))
    hmin=int(min(P[2],Q[2])-1*int(r))
    if hmin<0:
        hmin=0
    hmax=int(max(P[2],Q[2])+1*int(r)+1)
    
    x_range = np.sort(np.array([P[0], Q[0]]));
    y_range = np.sort(np.array([P[1], Q[1]]));
    z_range = np.sort(np.array([P[2], Q[2]]));
    #for any point on line x^2+y^2+z^2=<r^2
    for i in range(lmin,lmax):       
        for j in range (wmin,wmax):
            for k in range(hmin,hmax):
                p = np.array([i, j, k])
                AP = p - P;
                BP = p - Q
                proj_point = P + np.dot(AP,AB) * temp;
                distance_s = (proj_point[0] - p[0])**2 + (proj_point[1] - p[1])**2 + (proj_point[2] - p[2])**2;
                distance_1_s = (P[0] - p[0])**2 + (P[1] - p[1])**2 + (P[2] - p[2])**2;
                distance_2_s = (Q[0] - p[0])**2 + (Q[1] - p[1])**2 + (Q[2] - p[2])**2;

                if distance_s <=r*r and index_matrix[i,j,k]==0 and (x_range[0] <= proj_point[0] and proj_point[0] <= x_range[1]) and (
                            y_range[0] <= proj_point[1] and proj_point[1] <= y_range[1]) and (
                                    z_range[0] <= proj_point[2] and proj_point[2] <= z_range[1]) or distance_1_s<=r*r or distance_2_s<=r*r:
                    if index_matrix[i,j,k]==0:
                
                        index_matrix[i,j,k]=num_id
                if distance_s <=ri*ri and (x_range[0] <= proj_point[0] and proj_point[0] <= x_range[1]) and (
                            y_range[0] <= proj_point[1] and proj_point[1] <= y_range[1]) and (
                                    z_range[0] <= proj_point[2] and proj_point[2] <= z_range[1]) or distance_1_s<=ri*ri or distance_2_s<=ri*ri:
                    index_matrix[i,j,k]=255

                
#     end=time.time()
#     print (end-start,"s")
    return index_matrix#,xx,yy,zz


# In[15]:


#np.array([(length_1,radius[0],phi_1,theta_1,x[0],y[0],z[0],x[l],y[l],z[l]),(length_2,radius[1],phi_2,theta_2,x1[0],y1[0],z1[0],x1[l1],y1[l1],z1[l1])])
def s_airway_modeling0(index_matrix,s_branch,num_id=100):
#     start=time.time()
    P=np.array(s_branch)[4:7]+[360,360,0]
    Q=np.array(s_branch)[7:10]+[360,360,0]
    AB = Q - P;
    temp = AB / np.dot(AB,AB);
    r=s_branch[1]
    ri=0.6*r


    lmin=int(min(P[0],Q[0])-1.2*int(r))
    lmax=int(max(P[0],Q[0])+1.2*int(r))
    wmin=int(min(P[1],Q[1])-1.2*int(r))
    wmax=int(max(P[1],Q[1])+1.2*int(r))
    hmin=int(min(P[2],Q[2])-1*int(r))
    if hmin<0:
        hmin=0
    hmax=int(max(P[2],Q[2])+1*int(r)+1)
    
    x_range = np.sort(np.array([P[0], Q[0]]));
    y_range = np.sort(np.array([P[1], Q[1]]));
    z_range = np.sort(np.array([P[2], Q[2]]));
    #for any point on line x^2+y^2+z^2=<r^2
    for i in range(lmin,lmax):       
        for j in range (wmin,wmax):
            for k in range(hmin,hmax):
                p = np.array([i, j, k])
                AP = p - P;
                BP = p - Q
                proj_point = P + np.dot(AP,AB) * temp;
                distance_s = (proj_point[0] - p[0])**2 + (proj_point[1] - p[1])**2 + (proj_point[2] - p[2])**2;
                distance_1_s = (P[0] - p[0])**2 + (P[1] - p[1])**2 + (P[2] - p[2])**2;
                distance_2_s = (Q[0] - p[0])**2 + (Q[1] - p[1])**2 + (Q[2] - p[2])**2;

#                 if distance_s <=r*r or distance_t1<=r*r or distance_t2<=r*r and index_matrix[i,k,k]==0:
                if distance_s <=r*r and index_matrix[i,j,k]==0 and (x_range[0] <= proj_point[0] and proj_point[0] <= x_range[1]) and (
                            y_range[0] <= proj_point[1] and proj_point[1] <= y_range[1]) and (
                                    z_range[0] <= proj_point[2] and proj_point[2] <= z_range[1]) or distance_1_s<=r*r or distance_2_s<=r*r:
                    
                
                    index_matrix[i,j,k]=num_id

                if distance_s <=ri*ri and (x_range[0] <= proj_point[0] and proj_point[0] <= x_range[1]) and (
                            y_range[0] <= proj_point[1] and proj_point[1] <= y_range[1]) and (
                                    z_range[0] <= proj_point[2] and proj_point[2] <= z_range[1]) or distance_1_s<=ri*ri or distance_2_s<=ri*ri:
                    index_matrix[i,j,k]=255
#     end=time.time()
#     print (end-start,"s")
    return index_matrix#,xx,yy,zz

def s_airway_modeling(index_matrix,s_branch):
    index_matrix=s_airway_modeling0(index_matrix,s_branch[0])
    index_matrix=s_airway_modeling0(index_matrix,s_branch[1])
    return index_matrix


# In[16]:


def ss_airway_modeling(index_matrix,ss_branch):
    for i in range(0,4):
        index_matrix=s_airway_modeling0(index_matrix,ss_branch[i])
    return index_matrix


# ## Generating testing image

# In[17]:


def generating_test_image(index_matrix):
    start=time.time()
    print ("generating main branch")
    index_matrix=airway_modeling(index_matrix,trachea_i,trachea)
    index_matrix=airway_modeling(index_matrix,rm_i,rm)
    index_matrix=airway_modeling(index_matrix,ru_i,ru)
    index_matrix=airway_modeling(index_matrix,br_i,br)
    index_matrix=airway_modeling(index_matrix,r45_i,r45)
    index_matrix=airway_modeling(index_matrix,r6_i,r6)
    index_matrix=airway_modeling(index_matrix,rl7_i,rl7)
    index_matrix=airway_modeling(index_matrix,rl_i,rl)
    index_matrix=airway_modeling(index_matrix,r910_i,r910)
    index_matrix=airway_modeling(index_matrix,r8_i,r8)
    index_matrix=airway_modeling(index_matrix,r9_i,r9)
    index_matrix=airway_modeling(index_matrix,r10_i,r10)
    index_matrix=airway_modeling(index_matrix,r4_i,r4)
    index_matrix=airway_modeling(index_matrix,r5_i,r5)
    index_matrix=airway_modeling(index_matrix,r7_i,r7)
    index_matrix=airway_modeling(index_matrix,r1_i,r8)
    index_matrix=airway_modeling(index_matrix,r2_i,r2)
    index_matrix=airway_modeling(index_matrix,r3_i,r3)
    index_matrix=airway_modeling(index_matrix,lm_i,lm)
    index_matrix=airway_modeling(index_matrix,ll6_i,ll6)
    index_matrix=airway_modeling(index_matrix,l6_i,l6)
    index_matrix=airway_modeling(index_matrix,ll_i,ll)
    index_matrix=airway_modeling(index_matrix,l8_i,l8)
    index_matrix=airway_modeling(index_matrix,l910_i,l910)
    index_matrix=airway_modeling(index_matrix,l123_i,l123)
    index_matrix=airway_modeling(index_matrix,l12_i,l12)
    index_matrix=airway_modeling(index_matrix,l1_i,l1)
    index_matrix=airway_modeling(index_matrix,l2_i,l2)
    index_matrix=airway_modeling(index_matrix,l3_i,l3)
    index_matrix=airway_modeling(index_matrix,l45_i,l45)
    index_matrix=airway_modeling(index_matrix,l4_i,l4)
    index_matrix=airway_modeling(index_matrix,l5_i,l5)
    index_matrix=airway_modeling(index_matrix,lu_i,lu)
    index_matrix=airway_modeling(index_matrix,l9_i,l9)
    index_matrix=airway_modeling(index_matrix,l10_i,l10)
    
    print ("generating son branch")
    index_matrix=s_airway_modeling(index_matrix,sr1)
    index_matrix=s_airway_modeling(index_matrix,sr2)
    index_matrix=s_airway_modeling(index_matrix,sr3)
    index_matrix=s_airway_modeling(index_matrix,sr4)
    index_matrix=s_airway_modeling(index_matrix,sr5)
    index_matrix=s_airway_modeling(index_matrix,sr6)
    index_matrix=s_airway_modeling(index_matrix,sr7)
    index_matrix=s_airway_modeling(index_matrix,sr8)
    index_matrix=s_airway_modeling(index_matrix,sr9)
    index_matrix=s_airway_modeling(index_matrix,sr10)
    index_matrix=s_airway_modeling(index_matrix,sl1)
    index_matrix=s_airway_modeling(index_matrix,sl2)
    index_matrix=s_airway_modeling(index_matrix,sl3)
    index_matrix=s_airway_modeling(index_matrix,sl4)
    index_matrix=s_airway_modeling(index_matrix,sl5)
    index_matrix=s_airway_modeling(index_matrix,sl6)
    index_matrix=s_airway_modeling(index_matrix,sl8)
    index_matrix=s_airway_modeling(index_matrix,sl9)
    index_matrix=s_airway_modeling(index_matrix,sl10)
    
    print("generating grandson branch")
    index_matrix=ss_airway_modeling(index_matrix,ssr1)
    index_matrix=ss_airway_modeling(index_matrix,ssr2)
    index_matrix=ss_airway_modeling(index_matrix,ssr3)
    index_matrix=ss_airway_modeling(index_matrix,ssr4)
    index_matrix=ss_airway_modeling(index_matrix,ssr5)
    index_matrix=ss_airway_modeling(index_matrix,ssr6)
    index_matrix=ss_airway_modeling(index_matrix,ssr7)
    index_matrix=ss_airway_modeling(index_matrix,ssr8)
    index_matrix=ss_airway_modeling(index_matrix,ssr9)
    index_matrix=ss_airway_modeling(index_matrix,ssr10)
    index_matrix=ss_airway_modeling(index_matrix,ssl1)
    index_matrix=ss_airway_modeling(index_matrix,ssl2)
    index_matrix=ss_airway_modeling(index_matrix,ssl3)
    index_matrix=ss_airway_modeling(index_matrix,ssl4)
    index_matrix=ss_airway_modeling(index_matrix,ssl5)
    index_matrix=ss_airway_modeling(index_matrix,ssl6)
    index_matrix=ss_airway_modeling(index_matrix,ssl8)
    index_matrix=ss_airway_modeling(index_matrix,ssl9)
    index_matrix=ss_airway_modeling(index_matrix,ssl10)
    print ("swaping axes")
    index_matrix=np.array(index_matrix,dtype="f")
    index_matrix=np.swapaxes(index_matrix,0,2).copy()
    end=time.time()
    print(end-start,'s')
    return index_matrix


# In[19]:


def image_generation(index_matrix):
    index_matrix=generating_test_image(index_matrix)
    print ("generating Image")
    image=itk.GetImageFromArray(index_matrix)
    return image


# In[20]:


image=image_generation(index_matrix)


# In[21]:


view(image)


# In[ ]:




