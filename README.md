# IEPPCS-RUC-midterm

## 大学生创新实验计划和科研机基项目中期无人机影像判别模型代码

* 环境

  * Ubuntu 18.04
  * python 3.7.9
  * pytorch 1.6.0

* 数据集

  * 点击[这里](https://www.baidu.com)下载我们的训练集(提取码);
  * 将训练集下的三个文件夹`train`,`test`和`extra`放入`dataset`目录下.

* 训练

  * ```python
    python train.py
    ```

* 测试

  * ```
    python test.py
    ```
    
    这将输出在固体废弃物和非固体废弃物测试集上的准确率. 同时对`extra`中的每一张图像进行滑窗判定输出判定的结果到`dataset/result`下.
  
* 注意事项

  * 关于`GPU`, 如果电脑上有可用的`GPU`, 请修改`cfg.py`中的`USING_GPU`变量. 这个变量是一个python`list`类型, 请将其修改为当前设备可用的`GPU`设备号.

------

目前进行固体废弃物的识别的模型为`Resnet101`,  是一个具有101层结构的神经网络. 使用了`SGD`优化器和余弦下降的学习率. 后续还将尝试使用更多种类的神经网络.