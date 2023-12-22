import numpy as np


def build_norm_instance(num, maximum, f):
  for i in range(num):
    print(float(np.random.choice(np.arange(1,maximum+1))), file=f)
    # print(round(np.random.uniform(1,maximum),4), file=f)
    
num = 10000
for i in [8]:
  with open('data/psk_' + str(i) +  '_' + str(num) + '_discrete.txt', 'w') as f:
    build_norm_instance(num,i,f)
