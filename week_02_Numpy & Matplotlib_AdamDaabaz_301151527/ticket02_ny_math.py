Python 3.10.2 (tags/v3.10.2:a58ebcc, Jan 17 2022, 14:12:15) [MSC v.1929 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
import numpy as np
a = np.array([[5, 3, 1], [2, 4, 6]])
a
array([[5, 3, 1],
       [2, 4, 6]])
b.ndim
Traceback (most recent call last):
  File "<pyshell#3>", line 1, in <module>
    b.ndim
NameError: name 'b' is not defined
a.ndim
2
a=np.linspace(5,1,10)
a
array([5.        , 4.55555556, 4.11111111, 3.66666667, 3.22222222,
       2.77777778, 2.33333333, 1.88888889, 1.44444444, 1.        ])
np.shape(a)
(10,)
b=np.arange(250)
b
array([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,
        13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,
        26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,
        39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,
        52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,
        65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,
        78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,
        91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103,
       104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
       117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
       130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142,
       143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155,
       156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168,
       169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181,
       182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194,
       195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207,
       208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220,
       221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233,
       234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246,
       247, 248, 249])
c=np.ones((5,5))
c
array([[1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1.]])
d=np.zeros((7,3))
d
array([[0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.]])
e=np.eye(60
         )
e
array([[1., 0., 0., ..., 0., 0., 0.],
       [0., 1., 0., ..., 0., 0., 0.],
       [0., 0., 1., ..., 0., 0., 0.],
       ...,
       [0., 0., 0., ..., 1., 0., 0.],
       [0., 0., 0., ..., 0., 1., 0.],
       [0., 0., 0., ..., 0., 0., 1.]])
f=np.diag(np.array([1,1,1]))
f
array([[1, 0, 0],
       [0, 1, 0],
       [0, 0, 1]])
aa=np.random.rand(100)
aa
array([2.49593302e-01, 6.45381593e-01, 9.31096460e-01, 4.39621972e-01,
       4.53693114e-01, 5.66151055e-02, 8.96727610e-01, 6.90820753e-01,
       6.38299573e-01, 9.71240169e-01, 3.48756904e-01, 8.24942834e-01,
       6.54776205e-01, 4.35490725e-01, 4.03879334e-01, 3.42584696e-01,
       4.53617704e-01, 8.98401176e-01, 5.51275607e-01, 3.92047604e-01,
       7.06800530e-01, 3.89493744e-01, 2.54730371e-01, 2.23471983e-01,
       9.63576417e-01, 3.48212173e-01, 7.66679837e-01, 5.18451455e-01,
       6.00513542e-01, 5.81881621e-01, 4.46882511e-01, 7.47875371e-01,
       9.84504479e-01, 9.09729403e-01, 8.00975762e-01, 8.43212549e-01,
       8.40658757e-01, 5.08893007e-01, 1.24588851e-01, 9.97679503e-01,
       8.57576373e-01, 2.43400980e-01, 6.47877956e-01, 7.44523460e-01,
       1.65220414e-01, 5.46227779e-01, 5.01644110e-01, 9.19106778e-01,
       6.20063323e-02, 9.62583506e-01, 3.68175912e-01, 7.31264732e-01,
       9.61833879e-01, 7.29459668e-01, 7.68950695e-01, 5.74929649e-01,
       3.14511669e-01, 5.95430298e-01, 3.69276298e-01, 7.16947160e-01,
       9.01096835e-02, 8.57563612e-01, 7.61422539e-01, 5.07512581e-01,
       1.04429231e-02, 3.67460513e-01, 1.57193324e-01, 6.75565459e-01,
       3.94207866e-01, 7.26050965e-01, 7.17236367e-04, 6.07698514e-01,
       6.39576305e-02, 9.78799143e-01, 8.16428764e-01, 1.41713483e-01,
       3.92913941e-01, 5.62837009e-01, 9.86768234e-01, 1.80989188e-01,
       5.74721318e-01, 6.27425859e-01, 8.12222448e-01, 7.43240209e-01,
       5.28767721e-01, 3.79210409e-02, 4.01136665e-01, 4.86348462e-01,
       7.77425197e-01, 5.23606583e-01, 6.35043651e-01, 6.50329857e-01,
       1.60295193e-01, 2.76623387e-01, 3.96061206e-01, 7.62570437e-02,
       9.03579688e-01, 5.73650221e-03, 2.57874117e-01, 1.51017715e-01])
ab=np.random.rand(10)
ab
array([0.22238163, 0.31976976, 0.41380151, 0.86574359, 0.52504551,
       0.51147721, 0.17699083, 0.3298446 , 0.27106734, 0.24326976])
np.random.seed(5)
ac.random.rand(2)
Traceback (most recent call last):
  File "<pyshell#24>", line 1, in <module>
    ac.random.rand(2)
NameError: name 'ac' is not defined. Did you mean: 'a'?
ac=np.random.rand(3)
ac
array([0.22199317, 0.87073231, 0.20671916])
ac.np.empty
Traceback (most recent call last):
  File "<pyshell#27>", line 1, in <module>
    ac.np.empty
AttributeError: 'numpy.ndarray' object has no attribute 'np'
ac=np.empty
ac
<built-in function empty>
import timeit
setup='''
import numpy as np
R =range(50)
ad=np.arange(50)
'''
timeit.timeit("[i+5 for i in R]", setup)
1.5862858000000415
af = np.array([ 9, 7, 5, 3, 1, 0, 2, 4, 6, 8, 10])
af
array([ 9,  7,  5,  3,  1,  0,  2,  4,  6,  8, 10])
af=  np.array([ 9, 7, 5, 3, 1, 0, 2, 4, 6, 8)
              
SyntaxError: closing parenthesis ')' does not match opening parenthesis '['
af=  np.array([ 9, 7, 5, 3, 1, 0, 2, 4, 6, 8])
              
af.reshape((5,2))
              
array([[9, 7],
       [5, 3],
       [1, 0],
       [2, 4],
       [6, 8]])
af.reshape((2,5))
              
array([[9, 7, 5, 3, 1],
       [0, 2, 4, 6, 8]])
b66[0,2:4]
              
Traceback (most recent call last):
  File "<pyshell#39>", line 1, in <module>
    b66[0,2:4]
NameError: name 'b66' is not defined
af[0,2:4]
              
Traceback (most recent call last):
  File "<pyshell#40>", line 1, in <module>
    af[0,2:4]
IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed
af[0,1:2]
              
Traceback (most recent call last):
  File "<pyshell#41>", line 1, in <module>
    af[0,1:2]
IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed
af[1,2:4]
              
Traceback (most recent call last):
  File "<pyshell#42>", line 1, in <module>
    af[1,2:4]
IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed
af[0,1:4]
              
Traceback (most recent call last):
  File "<pyshell#43>", line 1, in <module>
    af[0,1:4]
IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed
ae=np.arange(0,50)
              
ae
              
array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
       34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49])
a36.reshape((5,10))
              
Traceback (most recent call last):
  File "<pyshell#46>", line 1, in <module>
    a36.reshape((5,10))
NameError: name 'a36' is not defined
ae.reshape((5,10))
              
array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
       [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
       [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
       [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]])
ae[0,3:5]
              
Traceback (most recent call last):
  File "<pyshell#77>", line 1, in <module>
    ae[0,3:5]
IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed
aa= ae.reshape((5,10))
              
aa
              
array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
       [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
       [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
       [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]])
aa[0,3:5]
              
array([3, 4])
aa[2,1:5]
              
array([21, 22, 23, 24])
aa[3,3:]
              
array([33, 34, 35, 36, 37, 38, 39])
aa[3::2]
              
array([[30, 31, 32, 33, 34, 35, 36, 37, 38, 39]])
ba=np.arange(10)
              
ba
              
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
np.sin(ba)
              
array([ 0.        ,  0.84147098,  0.90929743,  0.14112001, -0.7568025 ,
       -0.95892427, -0.2794155 ,  0.6569866 ,  0.98935825,  0.41211849])
np.log(ba)
              

Warning (from warnings module):
  File "<pyshell#87>", line 1
RuntimeWarning: divide by zero encountered in log
array([      -inf, 0.        , 0.69314718, 1.09861229, 1.38629436,
       1.60943791, 1.79175947, 1.94591015, 2.07944154, 2.19722458])
bb=np.eye(5)
              
b
              
array([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,
        13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,
        26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,
        39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,
        52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,
        65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,
        78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,
        91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103,
       104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
       117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
       130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142,
       143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155,
       156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168,
       169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181,
       182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194,
       195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207,
       208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220,
       221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233,
       234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246,
       247, 248, 249])
bb
              
array([[1., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0.],
       [0., 0., 1., 0., 0.],
       [0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 1.]])
np.cos(bb)
              
array([[0.54030231, 1.        , 1.        , 1.        , 1.        ],
       [1.        , 0.54030231, 1.        , 1.        , 1.        ],
       [1.        , 1.        , 0.54030231, 1.        , 1.        ],
       [1.        , 1.        , 1.        , 0.54030231, 1.        ],
       [1.        , 1.        , 1.        , 1.        , 0.54030231]])
np.exp(bb)
              
array([[2.71828183, 1.        , 1.        , 1.        , 1.        ],
       [1.        , 2.71828183, 1.        , 1.        , 1.        ],
       [1.        , 1.        , 2.71828183, 1.        , 1.        ],
       [1.        , 1.        , 1.        , 2.71828183, 1.        ],
       [1.        , 1.        , 1.        , 1.        , 2.71828183]])
np.triu(bb)
              
array([[1., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0.],
       [0., 0., 1., 0., 0.],
       [0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 1.]])
np.diag(ba)
              
array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 4, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 5, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 6, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 7, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 8, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 9]])
np.sum(ba)
              
45
bb[2,:].min()
              
0.0
ba[0,:].max()
              
Traceback (most recent call last):
  File "<pyshell#97>", line 1, in <module>
    ba[0,:].max()
IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed
bb[0,:].max()
              
1.0
bb.argmin()
              
1
bb
              
array([[1., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0.],
       [0., 0., 1., 0., 0.],
       [0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 1.]])
np.std(bb)
              
0.4
np.std(bb, axis =1)
              
array([0.4, 0.4, 0.4, 0.4, 0.4])
np.mean(bb)
              
0.2
np.median(bb)
              
0.0
np.all(bb != 0)
              
False

======================================================== RESTART: Shell ========================================================

=============================================== RESTART: C:/Users/adamd/chart.py ===============================================

=============================================== RESTART: C:/Users/adamd/chart.py ===============================================
