close all;clc
%生成200组输入样本集
N=200;
Ntrain=150;
x1_vec=linspace(-2,2,N)'; %产生以-2、2为始末的N个元素的行向量
x2_vec=unifrnd(-2,2,N,1); %生成（连续）均匀分布的随机数:产生一个N*1数组，数组中元素为-2~2的随机数
x3_vec=x1_vec;  
x4_vec=unifrnd(-2,2,N,1);
X_Data=[x1_vec, x2_vec, x3_vec,x4_vec]';%样本集矩阵形式
X0_M=X_Data(:,1:Ntrain);
X0_test=X_Data(:,Ntrain+1:N);
%得到期望输出
d1_vec=(x1_vec+x2_vec.^2+x3_vec.^3+x4_vec.^4)./(x1_vec.^2+x2_vec.^2+1);
d2_vec=(exp(-abs(x1_vec+x2_vec))+2)./(x3_vec.^2+x4_vec.^2+1);
%按输出的维数统计
D_Data=[d1_vec,d2_vec]';%期望值的矩阵形式按输出的时间统计
D0_M=D_Data(:,1:Ntrain);
D0_test=D_Data(:,Ntrain+1:N);

%Step2：初始化神经网络参数。
IALL=[9,9,9]; %第11层至Ir－1层的神经网络节点数
BPtype=[1,2,1,2]; %第11至r层的神经网络传递函数，1为 sigmoid tansig函数,2为tansig函数
r=length(BPtype);%得到神经网络的层数
maxstep=40000; %神经网络最大训练次数
threshold0=1e-2;%神经网络的精度阈值
etha0=1e-3;%神经网络的学习速率
mc=0;%附加动量因子
%将各隐层节点数、最大训练次数、精度阈值、学习速率、动量因子整合为一个参数
BPparameter={IALL, maxstep,threshold0,etha0,mc};
WIS_M=[];%初始连接权矩阵,可选参数,不输入时填[]
alpha0=1;%sigmoid 函数的参数，默认值为1

[WIS_M, En,maxXM_vec,minXM_vec, maxDM_vec,minDM_vec]=example30_2(X0_M,D0_M,BPtype,BPparameter, WIS_M, alpha0);
%Step3：调用多层神经网络函数 example30＿2解算相关参数，输出并显示神经网试情况，结果如图30－2和图30－3所示。
%神经网络函数输出的参数为
%WISM：各层的连接权矩阵
%En：迭代结束时的误差
%maxXM_vec,minXM_vec, maxDM_vec,minDM_vec：采样输入输出的最大与最小值向量，期望输出的最大与最小值向量
 
disp('经过解算得到各层的连接权矩阵如下')
for i=1: 4
    disp(['第I',num2str(i-1),'层至第num2str（i）层的连接权矩阵为：']);
    WIS_M{i}
 end
disp(['神经网络的误差为',num2str(En)]);

%step4:利用测试样本对神经网络进行测试
Y_M=example30_3(x0_test, BPtype, alpha0, WIS_M, maxXM_vec, minXM_vec, maxDM_vec, minDM_vec);
SampleNum=(Ntrain+1:N);%样本序号集
figure('name','神经网络的期望输出与实际输出的比较','numbertitle',off);
subplot(211)
plot(SampleNum, D0_test(1,: ), SampleNum,Y_M(1,: ),'linewidth',3);
grid on:
h=legend('期望输出','实际输出');
set(h,'fontsize',16);
title('利用测试样本得到神经网络的第一维期望输出与实际输出的比较','fontsize',16);
xlabel('样本序号',' fontsize',16);
ylabel('输出',' fontsize',16);
set(gca,'fontsize', 16); 
subplot(212)
plot(SampleNum, D0_test(2,: ), SampleNum,Y_M(2,: ),'linewidth',3);
grid on:
h=legend('期望输出','实际输出'); set(h,fontsize, 16);
set(h,'fontsize',16);
title('利用测试样本得到神经网络的第二维期望输出与实际输出的比较','fontsize',16);
xlabel('样本序号',' fontsize',16);
ylabel('输出',' fontsize',16);
set(gca,'fontsize', 16); subplot(212)

