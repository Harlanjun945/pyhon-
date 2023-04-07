#相关模块的引入
import os
import time
from qutip import *
import matplotlib as mp
import matplotlib.pyplot as plt
import numpy as np
from numpy import polyfit, poly1d
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from qutip.ipynbtools import version_table
import scipy.optimize as optimize
import seaborn as sns
import pandas as pd
import shelve as she
import shutil
from matplotlib.pyplot import MultipleLocator
import matplotlib.font_manager
from scipy.integrate import simpson
#-------------------------------------------常用函数---------------------------------------
#1.                                          保存数据
def save(name, data):
    def mkdir(path):
     
    	folder = os.path.exists(path)
     
    	if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
    		os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径
    
    
    mkdir('.\data')             #调用函数
    data = pd.DataFrame(data)
    data.to_excel('.\data\{}.xlsx'.format(name))		# ‘page_1’是写入excel的sheet名

#2.                                          读取数据
def load(name):
    sheet = pd.read_excel(io = '.\data\{}.xlsx'.format(name))
    to_array = np.array(sheet)
    data = np.delete(to_array, 0,axis = 1)
    if data.shape[1] == 1: #一维数据特殊处理
        data = data.T[0]
    print(data)
    return data

#3.                                           检索数据
def search():
    path = os.listdir('./data')
    data = [i for i in path if i.split('.')[1] == 'xlsx']
    for i in data:
        print(i.split('.')[0])
        
#4.                                           删除数据        
def delete(name):
    os.remove('.\data\{}.xlsx'.format(name))
#5.                                           布洛赫球
def bloch_d(result,lg0,lg1): #动态布洛赫
#result：mesolve函数返回的参数 
#lg0:布洛赫球态空间中的z轴量子态 ；lg1：布洛赫球态空间的-z轴量子态
    sx = lg0*lg1.dag() + lg1*lg0.dag()
    sz = lg0*lg0.dag() - lg1*lg1.dag()
    sy = 1j*(lg1*lg0.dag() - lg0*lg1.dag())
    b = Bloch()
    b.make_sphere()
    for i in np.arange(0, len(result.states)):
        x = expect(sx, result.states[i])
        y = expect(sy, result.states[i])
        z = expect(sz, result.states[i])
        b.add_vectors([x, y, z])
        b.show()
        time.sleep(0.01) 
        if i == len(result.states) - 1:
            pass
        else:
            b.clear()
def bloch_s(result, lg0, lg1): #静态布洛赫
    sx = lg0*lg1.dag() + lg1*lg0.dag()
    sz = lg0*lg0.dag() - lg1*lg1.dag()
    sy = 1j*(lg1*lg0.dag() - lg0*lg1.dag())
    b = Bloch()
    b.make_sphere()
    
    x = expect(sx, result.states)
    y = expect(sy, result.states)
    z = expect(sz, result.states)
    b.add_points([x, y, z])
    b.show()

#6.                                        拟合专用
def fit_fuction(x, y): #x,y为拟合数据， p0为尝试初解
    def f1(x,a):
        return 0.5*(1-np.cos(a*x))
    fs = np.fft.fftfreq(len(x), x[1] - x[0])
    Y = abs(np.fft.fft(y))
    freq = abs(fs[np.argmax(Y[1:]) + 1])
    a = 2 * np.pi * freq
    Omega = optimize.curve_fit(f1, x, y, p0=a, fmaxv = 10000)[0][0] 
    #拟合度
    y0 = [f1(i, Omega) for i in x] #带入得到拟合曲线
    res_ydata = np.array(y)-np.array(y0)
    ss_res = np.sum(res_ydata**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res/ss_tot) #拟合度计算
    result = [Omega, r_squared]
    print(r_squared)
    return result

































    
#---------------------------------------常用功能代码------------------------------------
counter = 0
# #图片
#图片格式设置
mp.rcParams['text.usetex'] = True
# del matplotlib.font_manager.weight_dict['roman']
# mp.font_manager._rebuild()
plt.rc('font',family='Times New Roman')
font = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 8,
        } #字体
plt.rcParams['font.sans-serif'] = ['Times New Roman'] #图片字体
plt.rcParams['axes.unicode_minus'] = False #防止乱码
plt.rc('legend', fontsize=16)
# plt.savefig('img_test.eps', dpi=600,format='eps', bbox_inches="tight")
# plt.show()

# #画图热力图专用
# fig,ax = plt.subplots(num=120, figsize=(3, 2), dpi=300) #图片大小和分辨率
# cet = plt.imshow(E, vmin=-200, vmax=20, cmap = 'bone')
# contour = plt.contour(x,y,E,[-190,-120,-60,-30,-10,0,12],linewidths = 1, colors = 'k')
# plt.clabel(contour, fontsize=8, inline = 'False', colors = 'k')
# plt.xticks([20, 49, 78],['-2', '0', '2'])
# plt.yticks([0, 49, 99],['-2', '0', '2'])
# plt.xlabel(r'$\Re \{ \alpha \} $', font)
# plt.ylabel(r'$\Im\{\alpha\}$', font)
# cb1 = plt.colorbar(cet, fraction=0.03, pad=0.1)
# tick_locator = mp.ticker.MaxNLocator(nbins=3)
# cb1.locator = tick_locator
# cb1.ax.set_title('E')
# cb1.set_ticks([0,-100,-200])
# cb1.update_ticks()
# plt.savefig('out.svg', dpi=300,format = 'svg', bbox_inches="tight")
# plt.show()


# #二维曲线图
# fig,ax = plt.subplots(num=None, figsize=(2.8, 1.7), dpi=300) #图片大小和分辨率
# ax.tick_params(labelsize=8) #坐标轴字体大小
# plt.plot(g_p[0], g_p[1],  linewidth = 1, color = '#E4392E', label = r'$a_{\theta} = 0, \ a_{\phi} = 0.1$')
# plt.plot(g_t[0], g_t[1],  linewidth = 1, color = '#3979F2', label = r'$a_{\theta} = 0.1, \ a_{\phi} = 0$')
# plt.xlabel(r'$\theta_0$', font)
# plt.ylabel(r'$\Omega$', font)
# x_major_locator=MultipleLocator(1.5)
# # #把x轴的刻度间隔设置为1，并存在变量里
# y_major_locator=MultipleLocator(0.03)
# #把y轴的刻度间隔设置为10，并存在变量里
# ax.xaxis.set_major_locator(x_major_locator)
# # #把x轴的主刻度设置为1的倍数
# ax.yaxis.set_major_locator(y_major_locator)
# #把y轴的主刻度设置为10的倍数
# plt.xlim((0, np.pi))
# plt.ylim((0, 0.06))
# plt.legend(prop = font,frameon = False)
# # plt.tight_layout()
# plt.savefig('out.svg', dpi=300,format = 'svg', bbox_inches="tight")
# plt.show()

# #计时
# start = time.time()
# end = time.time()
# print('运行了'+ str(end-start) + 's')


#--------------------------------------正文区-------------------------------------------





