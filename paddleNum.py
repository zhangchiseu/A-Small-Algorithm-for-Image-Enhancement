import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Linear
import numpy as np
import os
from PIL import Image

trainset = paddle.dataset.mnist.train()
train_reader = paddle.batch(trainset, batch_size=8)

class minist_model(fluid.dygraph.Layer):
    def __init__(self):
        super(minist_model,self).__init__()
        self.fc = Linear(input_dim=28*28,output_dim=1,act=None)
    def forward(self,inputs):
        outputs = self.fc(inputs)
        return outputs
model = minist_model()

with fluid.dygraph.guard():
    model = minist_model()
    model.train()
    train_loader = paddle.batch(paddle.dataset.mnist.train(),
                                batch_size=16)
    # 定义优化器
    opt = fluid.optimizer.SGDOptimizer(learning_rate=0.001,
                                       parameter_list=model.parameters())
"""
with fluid.dygraph.guard():
    model = minist_model()
    model.train()
    train_loader = paddle.batch(paddle.dataset.mnist.train(),
                                batch_size=16)
    # 定义优化器
    opt = fluid.optimizer.SGDOptimizer(learning_rate=0.001,
                                       parameter_list=model.parameters())
    for epoch_id in range(100):
        for batch_id, data in enumerate(train_loader()):
            image_data = np.array([x[0] for x in data]).astype('float32')
            label_data = np.array([x[1] for x in data]).astype('float32').reshape(-1, 1)
            image = fluid.dygraph.to_variable(image_data)
            label = fluid.dygraph.to_variable(label_data)

            pre = model(image)
            loss = fluid.layers.square_error_cost(pre, label)
            avg_loss = fluid.layers.mean(loss)
            if batch_id != 0 and batch_id % 1000 == 0:
                print("epoch: {}, batch: {}, loss is: {}".format(epoch_id, batch_id, avg_loss.numpy()))
            avg_loss.backward()
            opt.minimize(avg_loss)
            model.clear_gradients()
fluid.save_dygraph(model.state_dict(), 'mnist')
"""
import matplotlib.image as Img
import matplotlib.pyplot as plt



def load_image(img_path):

    im = Image.open(img_path).convert('L')
   # print(np.array(im))
    im = im.resize((28, 28), Image.ANTIALIAS)
    im = np.array(im).reshape(1, -1).astype(np.float32)
    # 图像归一化，保持和数据集的数据范围一致
    im = 1 - im / 127.5
    return im

# 定义预测过程
with fluid.dygraph.guard():
    model =minist_model()
    params_file_path = 'mnist'
    img_path = 'C:/Users/15729/Desktop/DIPtask2/Homework 2/ywj-a000_2.jpg'
# 加载模型参数
    model_dict, _ = fluid.load_dygraph("mnist")
    model.load_dict(model_dict)
# 灌入数据
    model.eval()#启动模型评价
    # 将一张图片转为一行向量
    tensor_img = load_image(img_path)
    print("数据集的大小为:",tensor_img.shape)
    result = model(fluid.dygraph.to_variable(tensor_img))
#  预测输出取整，即为预测的数字，打印结果
    print("本次预测的数字是", result.numpy().astype('int32'))
