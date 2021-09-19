#encoding = 'utf-8'

import os
from ctypes import *

folder = './hwrf'
dll = CDLL("colormapOptimization-上线版本1.dll")
print('ok')

# Declare c lang types
# 请传给我单精度float类型而不是double类型
FLOAT  	= c_float
INT 	= c_int
DOUBLE  = c_double
PFLOAT = POINTER(FLOAT)
PPFLOAT = POINTER(PFLOAT)

# convert python list to c 1D array
def pylist2carray(py_list, c_type_):
	c_array = (c_type_ * len(py_list))()
	for i in range(len(py_list)):
		c_array[i] = c_type_(py_list[i])
	return c_array


# 很简单的，反过来就行了
def carray2pylist(c_array):
	py_list = []
	for i in range(len(c_array)):
		py_list.append(c_array[i])
	return py_list


''' read data'''
rows = 0
cols = 0
datas = []
flag_col = True
flag_row = True

files = os.listdir(folder)
for fi in files:
	fi_d = os.path.join(folder,fi)
	if os.path.isfile(fi_d):
		with open(fi_d, 'r') as f:
			data = []
			for line in f.readlines():
				for w in line.split('\t'):
					w = w.strip()
					if len(w) != 0: 
						data.append(float(w))
						if flag_col == True:
							cols += 1
				flag_col = False
				if flag_row == True:
					rows += 1
			datas.append(data)
			flag_row = False


datalen = rows * cols

# Declare a 2d array
c_datas = (PFLOAT * len(datas))()
for i in range(len(datas)):
	c_datas[i] = (FLOAT * datalen)()
	for j in range(datalen):
		c_datas[i][j] = FLOAT(datas[i][j])


''' initialize readin data '''
dll.py_init_data(c_datas, len(datas), rows, cols)




''' initialize initial anchors' position and colors '''
anchorPos = [ 0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0 ]
anchorColor = [ 
	0.49996865, 8.8474626e-05, 0.99994153,
	0.24843583, 0.38286114, 0.98047245,
	0.0026615777, 0.70897233, 0.92190075,
	0.24476461, 0.91989356, 0.83201414,
	0.50418043, 1, 0.70311028,
	0.75400811, 0.91988468, 0.55090362,
	1, 0.6992749, 0.37695834,
	1, 0.38299969, 0.1956459,
	1, 3.1459182e-05, 0
	]



c_anchorPos = pylist2carray(anchorPos, FLOAT)
c_anchorColor = pylist2carray(anchorColor, FLOAT)
dll.py_init_anchors(c_anchorPos, c_anchorColor, 9)






''' fix position of given anchors '''
# anchorFixed = [ 3 ]

# c_anchorFixed = pylist2carray(anchorFixed, INT)
# dll.py_fix_anchors(c_anchorFixed, 1)







''' input roi contours '''
# x = [ 0 ]
# y = [ 0 ]

# c_x = pylist2carray(x, INT)
# c_y = pylist2carray(y, INT)
# dll.py_input_roi(c_x, c_y, 1) #


''' initialize balance factor '''
dll.py_init_param(FLOAT(10000), FLOAT(0), FLOAT(0), INT(100))



''' initialize background colors '''
bgColor = [ 0, 0, 0 ]

c_bgColor = pylist2carray(bgColor, FLOAT)
dll.py_init_bg(c_bgColor)



''' optimize anchors' position '''
dll.py_optimize_anchors(c_anchorPos)



resAnchorPos = carray2pylist(c_anchorPos)
print("py get it.")
print(resAnchorPos)