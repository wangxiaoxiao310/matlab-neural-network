%子函数 example30_3的目的是根据已经训练好的带偏置神经网络，计算样本输入神经网络后得到的输出；算法是利用神经网络的正向传播式（30－14）和结合反归一化式（30－34）计算神经网络输出，代码如下。
%输入变量说明
%％X_M样本输入
%BPtype 神经网络类型，1为sigmoid函数，2为tansig函数
%alpha0：神经网络的参数
%wism神经网络各层权值
%maxXM_vec,minXM_vec， maxDM_vec minDM_vec：已训练的神经网络中样本各维最大值与最小值边界，Y样本各维最大值与最小值边界
%输出变量说明
%Y_M：神经网络的输出
function Y_M=example30_3(X_M, BPtype, alpha0, WIS_M, maxXM_vec, minXM_vec, maxDM_vec, minDM_vec)
r=length(WIS_M);%计算神经网络的层数
I0=size(X_M,1)+1;%第0层输入数
N=size(X_M,2);%总的采样时间
beta=0.01;%归一化区间
VI_M{1}=[(X_M-minXM_vec*ones(1, N))./((maxXM_vec-minXM_vec)*ones(1,N))];%归一化输入
if isempty(alpha0==1)
    alpha0=l;
end
for s=O:r-1
    VI_M{s+2}=BPDirectionCal(BPtype(s+1), WIS_M{s+1},VI_M{s+1},N,alpha0); %it#＋1层的输出
end
%反归一化
Y_M=(VI_M{r+1}-beta).*((maxDM_vec-minDM_vec)*ones( 1, N))/(1-2*beta)+minDM_vec*ones(1,N);

end