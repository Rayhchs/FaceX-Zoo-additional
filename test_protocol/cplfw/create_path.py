from glob2 import glob

path = './croped'

imgfiles = glob(path+'/**/**.jpg')

with open('./img_list.txt', 'w+') as f:
	for i in imgfiles:
		f.write(i+'\n')