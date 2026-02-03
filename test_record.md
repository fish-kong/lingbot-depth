我测试了自己用realsense拍摄的的近景图片

![image](https://github.com/fish-kong/lingbot-depth/tree/dev/examples/8/rgb.jpg)

发现模型的深度值会变大，如下图所示，这是原始深度

![image](https://github.com/fish-kong/lingbot-depth/tree/dev/original.png)

点击树枝上的几个点，深度值普遍是0.31m

![image](https://github.com/fish-kong/lingbot-depth/tree/dev/refine.png)

同样点击树枝上的几个点，深度值从0.34到0.43

这样的深度值不利于我机械臂抓取的精度, 请问是否能够保留原始正确的值，补全缺失的值



