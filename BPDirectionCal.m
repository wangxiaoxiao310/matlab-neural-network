%BP神经网络的正向传播
function VI_M=BPDirectionCal(BPtypenum, WIS_M,VI_M,N, alpha0)
%本函数的目的是计算BP神经网络的正向传播
%BPtypenum sigmoid函数，2为 tansig函数
%WIS_M：s＋1层权值矩阵
%VI_M：s层隐层的输出
%N：样本数
% alpha0 sigmoid函数参数
switch BPtypenum
    case 1   %sigmoid函数
         VI_M=1./(1+exp(-alpha0*WIS_M*[VI_M;ones(1,N)]));%计算第s＋1层的输出
         %VI_M{s+2}＝1./(1+exp(-alpha0*WIS_M{s+1}*[VI_M{s+1};ones(1,N)]));％计算第s＋1层的输出
    case 2    %tansig函数
         VI_M=1-2./(1+exp(2*WIS_M'*[VI_M;ones(1,N)]));%计算第s＋1层的输出
         %VI_M{s+2}＝1-2./(1+exp(2*WIS_M'*WIS_M{s+1}*[VI_M{s+1};ones(1,N)]));％计算第s＋1层
    otherwise
         error('未知的神经网络类型')
end

