# TinyMind-start-with-0
从零开始深度学习：TinyMind汉字书法识别

操作步骤  
1. 从官网下载[数据集](http://www.tinymind.cn/competitions/41#dataDescription "数据集文件")，并解压到当前文件夹。产生train test1 两个文件  
2. 运行data.py文件，进行转录，将原始数据集转录为numpy矩阵，生成data.npy及label.npy  
3. 运行train.py进行训练  
4. 运行test.py使用训练完成的网络生成test.cvs文件上传[官网](http://www.tinymind.cn/competitions/41)进行测试

## 工程组织
data.py 数据转换文件  
train.py 网络训练文件  
model.py 网络描述文件  
test.py 最终测试结果生成文件


该项目旨在示范使用pytorch进行深度学习的大体过程，网络结构及超参数都是随意给的，抛砖引玉，欢迎提问！