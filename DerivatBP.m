%BP������ķ��򴫲���������
function fd_Is=DerivatBP(BPtypenum, VI_M,alpha0)
%��������Ŀ���Ǽ���BP������ķ��򴫲�����
%BPtypenum sigmoid������2Ϊ tansig����
%vIM��Is����������
%alpha0 sigmoid��������
%fd_Is���򴫲�����
switch BPtypenum
    case 1  %sigmoid����
        fd_Is=alpha0*VI_M'.*(1-VI_M');
    case 2 %tansig����
        fd_Is=(1+VI_M').*(1-VI_M');
    otherwise
    error('δ֪������������')
end

