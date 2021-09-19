//////////////////////////////////////////////////////////////////
Colormap-optimization CUDA version with Knitro
//////////////////////////////////////////////////////////////////

类组织：
cMap 	colormap的类，是一个分段线性函数。
cMapOpt 里面有计算目标函数的类
cMapUtils 各种工具函数，主要是辅助计算
myColor 计算颜色空间中的CIEDE2000和CIE76距离
stdafx  各种宏和常量。还有文件和颜色表名。
interface 提供了颜色表优化的函数接口。
main 主要是作测试用。



使用方法：
项目配置类型为“应用程序(.exe)”时，程序以main为入口运行。
项目配置类型为“动态库(.dll)”时，程序不会运行，而是生成以interface中函数为接口调用的dll。需要这些dll放入到./colormap-app-v5/C++中。


TODO:
knitro是商业优化库，识别机器码的，试用期只有三个月。