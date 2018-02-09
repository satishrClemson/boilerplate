#animals=['dog','car','cow']
##for idx, value in enumerate(animals):
##    print idx,value
#len_animals=(len(x) for x in animals)
##print(len_animals)
#
#
#class Polynomial:
#    def __init__(self, *coeff):
#        self.coeff=coeff
#
#    def __add__(self, other):
#        return Polynomial(*(x+y for x,y in zip(self.coeff, other.coeff)))
#    
#    def __repr__(self):
#        return 'Polynomial({})'.format(self.coeff)
#
#        
#p1=Polynomial(1,2,3)
#p2=Polynomial(3,4,5)
##print((p1+p2))
##print(p1.coeff)
#
#class Fib:
#    def __init__(self, max):
#        self.max = max
#        self.a = 0
#        self.b = 1
#
#    def __iter__(self):
#        return self
#
#    def next(self):
#        fib = self.a
#        if fib > self.max:
#            raise StopIteration
#        self.a, self.b = self.b, self.a + self.b
#        #print(id(fib),id(self.a),id(self.b))
#        return fib
#    
#
#
#def fib(max):
#    a,b=0,1
#    while a<max:
#        yield a
#        a,b=b,a+b
#
#
#gen=fib(10)
#print(next(gen))
#print(next(gen))
#print(next(gen))
#print(next(gen))
#
##for i in fib(10):
##        print(i)
#
#import numpy as np
#
#d=[1,2,3]
#d_slice=d[1:]
#d_slice[1]=2
##print(d)
##print(id(d))
##print(type(d))
#vData=[[1,2,3,4]]
##print(len(vData))
#aData=np.array(vData)
#aData_slice=aData[:,2:]
#aData_slice[:]=0
##print(aData)
##print(id(aData_slice))
##print(aData**2)
##print(aData.shape)
##print(aData.ndim)
#
#import os, glob
#import matplotlib
#metadata_dict={f:f.upper() for f in glob.glob('*')}
##print(metadata_dict.items())
##for f,meta in metadata_dict.items():
##    print(f,meta)
#        

############################# File handling and parsing Boilerplate Code ############################
import glob, os, argparse
parser = argparse.ArgumentParser()
parser.add_argument('--folder_path', type=str, help= 'Folder path')
FLAGS = parser.parse_args()
training_img_list_path = "/".join((FLAGS.folder_path,'training'))

if not os.path.exists(training_img_list_path):
    os.makedirs(training_img_list_path)

for fileName in glob.glob("/".join((FLAGS.folder_path,'bin/*.py'))):
    print(fileName)

exit()
############################ String Boilerplate Code ############################
s = ''
s_list = ['hello ','how ','are ','you ']
s = "".join(s_list)# Much more efficient since strings are immutable
print(s)
############################# Python Object Boilerplate Code ############################

class Person:
    def __init__(self,initialAge):
        if initialAge > 0:
            self.age=initialAge
        else:
            self.age=0
            print('Age is not valid, setting Age to 0')

    def yearPasses(self):
        self.age+=1
    
    def amIOld(self):
        if self.age < 13:
            print('You are young')
        if self.age >=13 and self.age <18:
            print('You are a teenager')
        else:
            print('You are old')


######################## Numpy Boilerplate Code############################
import numpy as np
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print(np.linalg.matrix_rank(a), a.shape[0], a.shape[1])
b = np.array([1,3,5,6])
bv = np.tile(b,(3,1))
print(a+bv)

################## Matplotlib Boilerplate Code (Needs testOpenCV virtual environment)  ##############
import matplotlib.pyplot as plt

x=np.arange(0,3*np.pi,0.1)
#y=np.sin(x)
#plt.plot(x,y)
#plt.show()

y_sin=np.sin(x)
y_cos=np.cos(x)
plt.style.use('ggplot')
plt.plot(x,y_sin)
plt.plot(x,y_cos)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Sine and Cosine')
plt.legend(['Sine','Cosine'])
plt.show()



######################## SciPy Boilerplate Code############################
from scipy.misc import imread, imsave, imresize
img = imread('../../Proposals/Low_light_Face_Recognition_SOCOM/method2pipeline.png')
print(img.shape)
img_tinted = img*[1,0.95,0.9,0.5]
plt.subplot(2,1,1)
plt.imshow(img)
plt.subplot(2,1,2)
plt.imshow(np.uint8(img_tinted))
plt.show()


######################## OpenCV Boilerplate Code############################
import cv2
img = cv2.imread('../../Proposals/NRO-Sensemaking/TPE.png')
img = cv2.resize(img, (img.shape[1]/2,img.shape[0]/2))
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
