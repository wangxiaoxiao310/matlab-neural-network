function [WIS_M, En,maxXM_vec,minXM_vec, maxDM_vec,minDM_vec]=example30_2(X_M,D_M,BPtype,BPparameter, WIS_M, alpha0)
%输入变量说明
%X_M：样本输入，（I0－1）*N矩阵，各行表示各维数的输入，各列表示各时间的输入
%DM期望输出，Ir*N矩阵，各行表示各维数的期望输出，各列表示各时间的期望输出
%BPtype sigmoid函数，2为 tansig函数
%BPparameter：神经网络的参数，依次为ILL，各隐层的节点数，如IALL＝［2，6，4］表示第一隐层，有2个节点，第二隐层有6个节点，第三隐层有4个节点
%maxstep threshold，神经网络的精度阅值；etha0，神经网络的学习速率
%mc，附加动量因子
%WIS_M：初始连接权矩阵，可选参数，不输入时填
%alpha0 sigmoid函数的参数，默认值为1
%输入格式： BPparameteralpha，iall，threshold，etha0｝；
%输出变量说明
%WIS_M:各层的连接权矩阵
%En迭代结束时的误差
%max_XM vec min XM vec， maxDM vec minDM vec：采样输入、输出的最大与最小值向量，期望输出的最大与最小值向量

%第一步，参数初始化
[IALL, maxstep, threshold0,etha0,mc]=deal(BPparameter{:});%得到各隐层节点数、最大迭代、步长、精度、学习速率(A=deal（B)函数把列表B的值一一赋值给A)
tic %开始计时
beta=0.01;%归一化参数，归一化区间至［beta，1－beta］
n=1;%计数器
I0=size(X_M,1);%第0层输入数，返回第二维即列数即样本数
[Ir,N]=size(D_M);%激活函数的层数，总时长,size未指定第二个参数（指定返回维数）,则都返回,N为样本个数
if size(IALL,2)>1
    IALL=IALL';%如果是横向量，则变为纵向量
end

IALL=[I0;IALL;Ir];
%归一化样本输入
minXM_vec= min(X_M,[],2);maxXM_vec=max(X_M,[],2);%计算样本输入各维的最大与最小值
minDM_vec= min(D_M,[],2);maxDM_vec=max(D_M,[],2);%计算期望输出各维的最大与最小值
VI_M{1}=[(X_M-minXM_vec*ones(1, N))./((maxXM_vec-minXM_vec)*ones(1,N))];%归一化输入
H_M{1}=(D_M-minDM_vec*ones(1, N))./((maxDM_vec-minDM_vec)*ones(1,N))*(1-2*beta)+beta;%归一化输出
%计算网络层数
r=length(IALL)-1;
%计算BP神经网络激活函数是否每层都有
if length(BPtype)==1   %如果只输入一个数，则认为每层都是这个激活函数
     BPtype=ones(1,r);
else
    if length(BPtype)~=r
        error(['激活函数的个数为',num2str(length(BPtype)),'个，而神经网络的隐层数为',num2str(r),'个，二者不匹配.']);
    end
end
numP=0;%计算神经网络中待求权值数量
 for s=1: r
    numP=numP+(IALL(s)+1)*IALL(s+1);
 end
disp(['BP神经网络共有',num2str(numP),'个待求的权值：',num2str(10),'x',num2str(N),'维输入:',num2str(Ir), 'x',num2str(N),'维输出.'])
disp(['BP神经网络I1至',num2str(r),'层的激活函数情况：'])
disp(['第 ',num2str(find(BPtype==1)),'层为 sigmoid激活函数：第',num2str(find(BPtype==2)),'层为 tansig激活函数'])
%记录总误差
REA_vec=[];
if isempty(alpha0)==1 %如果 alpha0输入为空集，则令其为1
     alpha0=l;
end
if isempty(WIS_M)==1 %如果不输入初始连接矩阵，则随机生成初始连接矩阵
    for s=0:r-1
        WIS_M{s+1}=normand(0,1,IALL(s+1),IALL(s+2));%计算第s＋1层的连接权矩阵
        VI_M{s+2}=BPDirectionCal(BPtype(s+1), WIS_M{s+1},VI_M{s+1},N,alpha0); %计算 s+1层的输出
     end
else
    for s=0:r-1
        VI_M{s+2}=BPDirectionCal(BPtype(s+1), WIS_M{s+1},VI_M{s+1},N,alpha0); %计算 s+1层的输出

    end
end
DeltaWIS_M=WIS_M;%令初始权值增量矩阵等于权值矩阵
%计算总误差
EA_M=H_M-VI_M{r+1};
En=norm(EA_M, 'fro')/sqrt(2);%矩阵f-范数（绝对值平方和）
%第二步，利用本章提出的多层神经网络迭代方法计算神经网络的权值矩阵及误差
while En>threshold0 && n<maxstep
    GammaIr_M{1}=DerivatBP(BPtype(r),VI_M{r+1},alpha0).*EA_M';%计算层的输出
    DeltaWIS_M{r}=mc*DeltaWIS_M{r}+(1-mc)*etha0*[VI_M{r}; ones(1, N)]*GammaIr_M{1};%利用附加动量法计算权值增量
    WIS_M{r}=WIS_M{s}+DeltaWIS_M{r};%计算层连接权矩阵的修正
    % BP神经网络误差的反向传播
    for s=r-1:-1: 1
         %计算s层的输出
         GammaIr_M{r+l-s}=DerivatBP(BPtype(s),VI_M{s+1},alpha0).*(GammaIr_M{r-s}*WIS_M{s+1}(1:ALL(s+1),:)');
         %利用附加动量法计算权值增量
         DeltaWIS_M{s}=mc*DeltaWIS_M{s}+(1-mc)*etha0*[VI_M{s};ones(1,N)]*GammaIr_M{r+1-s};
         %计算Is层连接权矩阵的修正
         WIS_M{s}=WIS_M{s}+DeltaWIS_M{s};
    end
    n=n+1;
    %计数器加1
    %计算神经网络的实际输出
    for s=0: r-1
        VI_M{s+2}=BPDirectionCal(BPtype(s+1), WIS_M{s+1},VI_M{s+1},N,alpha0); %计算 s+1层的输出
    end
    %计算总误差
    EA_M=H_M-VI_M{r+1};
    En=norm(EA_M,'fro')/sqrt(2);
    REA_vec=[REA_vec;En];
 end
%第三步，保存并显示结果
%将训练好的结果保存到文件BPmatrixmat中
save('BPmatrix.mat',' WIS_M', 'maxXM_vec','minXM_vec',' maxDMvec','minDM_vec')
t=toc;%结束计时
t1=1:n-1;%因为n多加了一次，所以要减1
figure ('name',['BP神经网络的仿真时间为',num2str(t0),'秒'],'numbertitle','off')
plot(tl, REA_vec,' linewidth', 1.5);
grid on;
xlabel('迭代次数',' fontsize',16);
ylabel('误差',' fontsize',16);
title('带偏置的BP神经网络误差随迭代步数变化的情况',' fontsize',16);
set(gca,'fontsize', 16); 
%反归一化
Y_M=(VI_M{r+1}-beta).*((maxDM_vec-minDM_vec)*ones(1, N))./(1-2*beta)+minDM_vec*ones(1,N);
t2=1:N;
%作图
for s0=1: ceil(Ir/2)
    if s0<ceil(Ir/2)||mod(lr,2)==0
        figure('name',' BP神经网络的拟合情况 ','numbertitle','off')
        for sl=1: 2
            subplot(2,1,sl);
            plot(t2,D_M(2*(s0-1)+s1, :),t2,Y_M(2*(s0-1)+s1,:) ,'linewidth', 1.5);
            grid on;
            legend('期望输出','BP神经网络输出',16)
            xlabel('采样时间',' fontsize',16);
            ylabel(['第',num2str(2*(s0-1)+s1),'维输出'],' fontsize',16);
            title('期望输出与BP神经网络输出比较',' fontsize',16);
            set(gca,'fontsize', 16); 
        end
    else
      figure('name',' BP神经网络的拟合情况 ','numbertitle','off')
      plot(t2,D_M(Ir,:),t2,Y_M(Ir,:) ,'linewidth', 1.5);
      grid on;
     legend('期望输出','BP神经网络输出',16)
     xlabel('采样时间',' fontsize',16);
     ylabel(['第',num2str(Ir),'维输出'],' fontsize',16);
     title('期望输出与BP神经网络输出比较',' fontsize',16);
      set(gca,'fontsize', 16); 
   end
end





end

