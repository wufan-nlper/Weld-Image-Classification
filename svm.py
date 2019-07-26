import numpy as np
import os
import re
import sklearn
from sklearn.svm import SVC

path_of_TRAINDATA = '/home/wufan/Desktop/缺陷图像集/VectorsForTrain/'
path_of_TESTDATA = '/home/wufan/Desktop/缺陷图像集/VectorsForTest/'
#四个字段的正则表达式
bar_defect = r'\u6761\u5f62\u7f3a\u9677'
unpenetrated = r'\u672a\u710a\u900f'
unfused = r'\u672a\u7194\u5408'
circular_defect = r'\u5706\u5f62\u7f3a\u9677'
vectors = []	#预定存储图像转换的向量
labels = []	#预定存储图像的标签
x_test = []	#测试集向量
y_test = []	#测试集标签
for root, dirs, files in os.walk(path_of_TRAINDATA):	
	for file in files:		
		f = open(path_of_TRAINDATA + file, 'r')	#打开txt文件，只读
		lines = f.readlines()
		for line in lines:
			vec = line.split()	#将txt文件里按空格分开的数字存入数组vec
			a = []
			for ele in vec:
				ele = float(ele)	#将数组vec里的string型数字改为float
				a.append(ele)
		vectors.append(a)
		#用正则表达式匹配来输入标签数组
		if re.search(re.compile(bar_defect), file):	#条形缺陷记作0
			labels.append(0)
		elif re.search(re.compile(unpenetrated), file):	#未焊透记作1
			labels.append(1)
		elif re.search(re.compile(unfused), file):	#未熔合记作2
			labels.append(2)
		elif re.search(re.compile(circular_defect), file):	#圆形缺陷记作3
			labels.append(3)
		else:
			labels.append(9)
filename = []	#将测试数据的文件名记录，与分类结果一起打印输出
#将测试数据存入数组x_test
for root, dirs, files in os.walk(path_of_TESTDATA):
	for file in files:
		f = open(path_of_TESTDATA + file, 'r')
		lines = f.readlines()
		for line in lines:
			vec = line.split()	
			a = []
			for ele in vec:
				ele = float(ele)	
				a.append(ele)
		filename.append(file)
		x_test.append(a)
#将实际的标签存入数组y_test
for file in filename:
	if re.search(re.compile(bar_defect), file):	#条形缺陷记作0
		y_test.append(0)
	elif re.search(re.compile(unpenetrated), file):	#未焊透记作1
		y_test.append(1)
	elif re.search(re.compile(unfused), file):	#未熔合记作2
		y_test.append(2)
	elif re.search(re.compile(circular_defect), file):	#圆形缺陷记作3
		y_test.append(3)
	else:						#随便写的
		y_test.append(9)
#支持向量机
Vectors = np.array(vectors)	#训练集向量，要转换成numpy.array型向量
Labels = np.array(labels)	#训练集标签
X_Test = np.array(x_test)	#测试集向量
Y_Test = np.array(y_test)	#测试集标签
clf = SVC(gamma = 'auto')	#分类面
clf.fit(Vectors, Labels)	#训练分类面，具体参数参考https://www.cnblogs.com/luyaoblog/p/6775342.html
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)
print(filename)		#打印测试集文件名
print(clf.predict(X_Test))	#打印预测结果
print('训练集精度：' + str(100 * clf.score(Vectors, Labels)) + '%')	#打印精度
print('测试集精度：' + str(100 * clf.score(X_Test, Y_Test)) + '%')


