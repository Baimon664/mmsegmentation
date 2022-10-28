import numpy as np

class_weight = np.array([0.36761542, 3.71231467, 29.65874266, 0.30827189, 2.23740685, 104.10830057, 6.62986244, 0.43679222, 0.54845459, 212.72077164, 125.3518259 ])

cap = np.clip(class_weight, 1, 999)

m = np.max(cap)

#y = mx+ c
#2 = m(212.720) + c
#1 = m + c
#m =  0.0047
#c = 0.9953

def linear(x):
    m = 0.0047
    c = 0.9953
    return (m*x) + c

cap = cap.tolist()
for i in range(len(cap)):
    cap[i] = linear(cap[i])
print(cap)

