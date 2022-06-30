import numpy as np
import matplotlib.pyplot as plt

image=np.random.rand(30,30)

#### Activity 1 (Figure_img1.png)
##plt.imshow(image)

## Activity 2 (Figure_img2.png)
##plt.imshow(image, cmap=plt.cm.Accent)

#### Activity 3 (Figure_img3.png)
##plt.imshow(image, cmap=plt.cm.hot)

#### Activity 4 (Figure_img4.png)
plt.imshow(image, cmap=plt.cm.Pastell)

plt.colorbar()

plt.show()
