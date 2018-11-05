%BP神经网络的反向传播导数计算
function fd_Is=DerivatBP(BPtypenum, VI_M,alpha0)
%本函数的目的是计算BP神经网络的反向传播导数
%BPtypenum sigmoid函数，2为 tansig函数
%vIM：Is层隐层的输出
%alpha0 sigmoid函数参数
%fd_Is反向传播导数
switch BPtypenum
    case 1  %sigmoid函数
        fd_Is=alpha0*VI_M'.*(1-VI_M');
    case 2 %tansig函数
        fd_Is=(1+VI_M').*(1-VI_M');
    otherwise
    error('未知的神经网络类型')
end

