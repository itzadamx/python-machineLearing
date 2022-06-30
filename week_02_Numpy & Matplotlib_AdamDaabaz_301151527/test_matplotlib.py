import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(0,7,20)
y=np.linspace(0,9,20)
## Activity 1 (Figure_1.png)
plt.plot(x,y)
plt.show()


## Activity 2 (Figure_1xy.png)
plt.plot(x,y, 'o')
plt.show()

## Activity 3 (Figure_1xy+1.png)
plt.plot(x+1,y+1, '_')
plt.show()


##Activity 4 (Figure_2.png)
import math
plt.plot(x,np.sin(x))
plt.show()
