import numpy as np
import os
import sys
import getopt
import caffe
#图片路径
image_path = '/home/wufan/Desktop/缺陷图像集/'
#caffe路径
caffe_root = '/caffe/'
#模型源文件
model_prototxt = caffe_root + 'models/bvlc_googlenet/deploy.prototxt'
#预训练好的模型
model_trained = caffe_root + 'models/bvlc_googlenet/bvlc_googlenet.caffemodel'
#包含类标签的文件
imagenet_labels = caffe_root + 'data/ilsvrc12/synset_words.txt'
#用于输入processing
mean_path = caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'
#抽取层的名字
layer_name = 'pool5/7x7_s1'
sys.path.insert(0, caffe_root + 'python')

#主函数
def main(argv):

	#使用CPU
	caffe.set_mode_cpu()
	
	#定义使用的神经网络模型
	net = caffe.Classifier(model_prototxt,model_trained,mean = np.load(mean_path).mean(1).mean(1),channel_swap = (2,1,0),raw_scale = 255,image_dims = (224,224))
	#将标签集合导入数组labels
	with open(imagenet_labels) as f:
    		labels = f.readlines()

	#对目标路径内的图像遍历并生成向量存储于对应的txt文件
	for root, dirs, files in os.walk(image_path):
		for dir in dirs:
			for subroot, subdir, images in os.walk(image_path + dir):
				for image in images:
					#加载要处理的图片
					image_file = os.path.join(subroot, image)
					input_image = caffe.io.load_image(image_file)
					#预测图片类别并打印
					prediction = net.predict([input_image], oversample = False)
					print(os.path.basename(image), ':', labels[prediction[0].argmax()].strip() , ' (', prediction[0][prediction[0].argmax()] , ')')
					#处理文件名，剪掉路径和后缀
					filepath, tmpfilename = os.path.split(image_file)
					shotname, extension = os.path.splitext(tmpfilename)
					txtname = root + 'Vectors/' + shotname + '.txt'
					#将向量存入文件夹，每张图片对应一个1024维向量
					np.savetxt(txtname, net.blobs[layer_name].data[0].reshape(1,-1), fmt='%.8g')
					#print(image_file)

if __name__ == "__main__":
	main(sys.argv)

		





















