Python 3.10.2 (tags/v3.10.2:a58ebcc, Jan 17 2022, 14:12:15) [MSC v.1929 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
import pandas as pd
series=pd.Series([1,2,3,4,5])
series
0    1
1    2
2    3
3    4
4    5
dtype: int64
series2=pd.Series([1,2,3,4,5], index=['a','b','c','d','e'])
series2
a    1
b    2
c    3
d    4
e    5
dtype: int64
series[3]
4
series2['d']
4
series.iloc[3]
4
series2.iloc[2]
3
dates1=pd.date_range('20210719',periods=12)
dates2=pd.date_range('20210719',periods=12,freq='Y')
dates3=pd.date_range('20210719',periods=12,freq='M')  #Monthly
series3=pd.Series([1,2,3,4,5,6,7,8,9,10,11,12])
series3.index=dates3
series3
2021-07-31     1
2021-08-31     2
2021-09-30     3
2021-10-31     4
2021-11-30     5
2021-12-31     6
2022-01-31     7
2022-02-28     8
2022-03-31     9
2022-04-30    10
2022-05-31    11
2022-06-30    12
Freq: M, dtype: int64
pd.read_csv(r'd:\test.csv')
Traceback (most recent call last):
  File "<pyshell#15>", line 1, in <module>
    pd.read_csv(r'd:\test.csv')
  File "C:\Users\adamd\AppData\Roaming\Python\Python310\site-packages\pandas\util\_decorators.py", line 311, in wrapper
    return func(*args, **kwargs)
  File "C:\Users\adamd\AppData\Roaming\Python\Python310\site-packages\pandas\io\parsers\readers.py", line 680, in read_csv
    return _read(filepath_or_buffer, kwds)
  File "C:\Users\adamd\AppData\Roaming\Python\Python310\site-packages\pandas\io\parsers\readers.py", line 575, in _read
    parser = TextFileReader(filepath_or_buffer, **kwds)
  File "C:\Users\adamd\AppData\Roaming\Python\Python310\site-packages\pandas\io\parsers\readers.py", line 933, in __init__
    self._engine = self._make_engine(f, self.engine)
  File "C:\Users\adamd\AppData\Roaming\Python\Python310\site-packages\pandas\io\parsers\readers.py", line 1217, in _make_engine
    self.handles = get_handle(  # type: ignore[call-overload]
  File "C:\Users\adamd\AppData\Roaming\Python\Python310\site-packages\pandas\io\common.py", line 789, in get_handle
    handle = open(
FileNotFoundError: [Errno 2] No such file or directory: 'd:\\test.csv'
pd.read_csv(r'd:\iris.csv')
Traceback (most recent call last):
  File "<pyshell#16>", line 1, in <module>
    pd.read_csv(r'd:\iris.csv')
  File "C:\Users\adamd\AppData\Roaming\Python\Python310\site-packages\pandas\util\_decorators.py", line 311, in wrapper
    return func(*args, **kwargs)
  File "C:\Users\adamd\AppData\Roaming\Python\Python310\site-packages\pandas\io\parsers\readers.py", line 680, in read_csv
    return _read(filepath_or_buffer, kwds)
  File "C:\Users\adamd\AppData\Roaming\Python\Python310\site-packages\pandas\io\parsers\readers.py", line 575, in _read
    parser = TextFileReader(filepath_or_buffer, **kwds)
  File "C:\Users\adamd\AppData\Roaming\Python\Python310\site-packages\pandas\io\parsers\readers.py", line 933, in __init__
    self._engine = self._make_engine(f, self.engine)
  File "C:\Users\adamd\AppData\Roaming\Python\Python310\site-packages\pandas\io\parsers\readers.py", line 1217, in _make_engine
    self.handles = get_handle(  # type: ignore[call-overload]
  File "C:\Users\adamd\AppData\Roaming\Python\Python310\site-packages\pandas\io\common.py", line 789, in get_handle
    handle = open(
FileNotFoundError: [Errno 2] No such file or directory: 'd:\\iris.csv'
pd.read_csv(r'C:\Users\adamd\AppData\Roaming\Python\Python310\site-packages\sklearn\datasets\data\iris.csv')
     150    4  setosa  versicolor  virginica
0    5.1  3.5     1.4         0.2          0
1    4.9  3.0     1.4         0.2          0
2    4.7  3.2     1.3         0.2          0
3    4.6  3.1     1.5         0.2          0
4    5.0  3.6     1.4         0.2          0
..   ...  ...     ...         ...        ...
145  6.7  3.0     5.2         2.3          2
146  6.3  2.5     5.0         1.9          2
147  6.5  3.0     5.2         2.0          2
148  6.2  3.4     5.4         2.3          2
149  5.9  3.0     5.1         1.8          2

[150 rows x 5 columns]
pd.read_csv(r'C:\Users\adamd\OneDrive\Desktop\ADAM COLLEGE\HW\COMP318\test3.txt')
        A B C
0     0 1 2 3
1     1 4 5 6
2     2 7 8 9
3  3 11 11 12
df_test=pd.read_csv(r'C:\Users\adamd\AppData\Roaming\Python\Python310\site-packages\sklearn\datasets\data')
Traceback (most recent call last):
  File "<pyshell#19>", line 1, in <module>
    df_test=pd.read_csv(r'C:\Users\adamd\AppData\Roaming\Python\Python310\site-packages\sklearn\datasets\data')
  File "C:\Users\adamd\AppData\Roaming\Python\Python310\site-packages\pandas\util\_decorators.py", line 311, in wrapper
    return func(*args, **kwargs)
  File "C:\Users\adamd\AppData\Roaming\Python\Python310\site-packages\pandas\io\parsers\readers.py", line 680, in read_csv
    return _read(filepath_or_buffer, kwds)
  File "C:\Users\adamd\AppData\Roaming\Python\Python310\site-packages\pandas\io\parsers\readers.py", line 575, in _read
    parser = TextFileReader(filepath_or_buffer, **kwds)
  File "C:\Users\adamd\AppData\Roaming\Python\Python310\site-packages\pandas\io\parsers\readers.py", line 933, in __init__
    self._engine = self._make_engine(f, self.engine)
  File "C:\Users\adamd\AppData\Roaming\Python\Python310\site-packages\pandas\io\parsers\readers.py", line 1217, in _make_engine
    self.handles = get_handle(  # type: ignore[call-overload]
  File "C:\Users\adamd\AppData\Roaming\Python\Python310\site-packages\pandas\io\common.py", line 789, in get_handle
    handle = open(
PermissionError: [Errno 13] Permission denied: 'C:\\Users\\adamd\\AppData\\Roaming\\Python\\Python310\\site-packages\\sklearn\\datasets\\data'
df_test=pd.read_csv(r'C:\Users\adamd\AppData\Roaming\Python\Python310\site-packages\sklearn\datasets\data\iris.csv')
df_test.describe()
              150           4      setosa  versicolor   virginica
count  150.000000  150.000000  150.000000  150.000000  150.000000
mean     5.843333    3.057333    3.758000    1.199333    1.000000
std      0.828066    0.435866    1.765298    0.762238    0.819232
min      4.300000    2.000000    1.000000    0.100000    0.000000
25%      5.100000    2.800000    1.600000    0.300000    0.000000
50%      5.800000    3.000000    4.350000    1.300000    1.000000
75%      6.400000    3.300000    5.100000    1.800000    2.000000
max      7.900000    4.400000    6.900000    2.500000    2.000000
import numpy as np
df1000=pd.DataFrame(np.random.randn(1000,4),columns=list('ABCD'))
df1000.describe()
                 A            B            C            D
count  1000.000000  1000.000000  1000.000000  1000.000000
mean     -0.028981    -0.067078     0.057031     0.033861
std       0.992232     0.995598     0.995807     0.989895
min      -3.110805    -3.150534    -3.243939    -3.854803
25%      -0.706255    -0.777610    -0.576471    -0.632876
50%      -0.049494    -0.054474     0.029301     0.028184
75%       0.616638     0.612824     0.731707     0.735435
max       3.050393     2.700595     3.465616     3.560784
df1000.mean()
A   -0.028981
B   -0.067078
C    0.057031
D    0.033861
dtype: float64
