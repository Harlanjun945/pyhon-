# 文档说明
二维曲线图 热力图 三维图的一些论文相关模板
## I.mode_code.py
### i.简介
  内含一些使用的函数：保存数据至excel表格 从excel表格导入数据到变量 搜索和编辑等
### ii.相关函数介绍
  **(a) save(name, data)**  
   *name输入保存的文件名 data输入变量名*
    该函数可以把python程序中定义的列表或矩阵数据变量，存入到excel表格内，并在当前py文件路径内新建一个data文件夹，把该excel表格用name.xlsx命名存储进该文件夹中。  
    例如，我们通过跑一个长程序，比如用ode45解复杂的微分方程组（假设这段代码跑了一晚上），最后得到一组x、y数据：  
    ![image](https://user-images.githubusercontent.com/130061058/230535564-d212ca95-3e1f-4310-9e97-d23c1c21d692.png)
![image](https://user-images.githubusercontent.com/130061058/230535573-01c4f1ab-8301-41da-9165-c61e1d225345.png)
    我们需要将这个数据以曲线图的形式展示出来，如图：  
    ![image](https://user-images.githubusercontent.com/130061058/230535946-766b6970-2edc-4008-a18c-81f151c05447.png)
    但在期刊投稿中，有时候需要对图片进行反复修改，而获取数据的时间成本很高，不可能每次都重新跑一次程序，获取数据再画图，因此需要该函数，将数据以excel表格形式保存，之后继续用python或者用matlab、origin等软件进行绘图都是十分方便的。
