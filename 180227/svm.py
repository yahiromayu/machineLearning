import numpy as np
import matplotlib.pyplot as plt

rng = np.random.RandomState(123)

d = 2
N = 10
mean = 5

x1 = rng.randn(N,d) + np.array([0,0])
x2 = rng.randn(N,d) + np.array([mean,mean])

#plt.plot(x1[:,0],x1[:,1],"o")
#plt.plot(x2[:,0],x2[:,1],"o")
#plt.show()

x = np.concatenate((x1,x2),axis=0)

w = np.zeros(d)
b = 0

def y(x):
	return step(np.dot(w,x) + b)

def step(x):
	return 1 * (x>0)

def t(i):
	if i<N:
		return 0
	else:
		return 1

def yy(x):
	return (-w[0,] * x - b) /w[1,]

while True:
	classfied = True
	for i in range(N*2):
		delta_w = (t(i) - y(x[i])) * x[i]
		delta_b = (t(i) - y(x[i]))
		w += delta_w
		b += delta_b
		classfied *= all(delta_w == 0) * (delta_b == 0)
	if classfied:
		break

tmpx = np.arange(-3,8,0.1)
tmpy = yy(tmpx)

plt.plot(x[:,0],x[:,1],"o")
plt.plot(tmpx,tmpy)
plt.show()
