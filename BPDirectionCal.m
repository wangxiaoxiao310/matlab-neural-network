%BP����������򴫲�
function VI_M=BPDirectionCal(BPtypenum, WIS_M,VI_M,N, alpha0)
%��������Ŀ���Ǽ���BP����������򴫲�
%BPtypenum sigmoid������2Ϊ tansig����
%WIS_M��s��1��Ȩֵ����
%VI_M��s����������
%N��������
% alpha0 sigmoid��������
switch BPtypenum
    case 1   %sigmoid����
         VI_M=1./(1+exp(-alpha0*WIS_M*[VI_M;ones(1,N)]));%�����s��1������
         %VI_M{s+2}��1./(1+exp(-alpha0*WIS_M{s+1}*[VI_M{s+1};ones(1,N)]));�������s��1������
    case 2    %tansig����
         VI_M=1-2./(1+exp(2*WIS_M'*[VI_M;ones(1,N)]));%�����s��1������
         %VI_M{s+2}��1-2./(1+exp(2*WIS_M'*WIS_M{s+1}*[VI_M{s+1};ones(1,N)]));�������s��1��
    otherwise
         error('δ֪������������')
end

