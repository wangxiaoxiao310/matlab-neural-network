%�Ӻ��� example30_3��Ŀ���Ǹ����Ѿ�ѵ���õĴ�ƫ�������磬�������������������õ���������㷨����������������򴫲�ʽ��30��14���ͽ�Ϸ���һ��ʽ��30��34������������������������¡�
%�������˵��
%��X_M��������
%BPtype ���������ͣ�1Ϊsigmoid������2Ϊtansig����
%alpha0��������Ĳ���
%wism���������Ȩֵ
%maxXM_vec,minXM_vec�� maxDM_vec minDM_vec����ѵ������������������ά���ֵ����Сֵ�߽磬Y������ά���ֵ����Сֵ�߽�
%�������˵��
%Y_M������������
function Y_M=example30_3(X_M, BPtype, alpha0, WIS_M, maxXM_vec, minXM_vec, maxDM_vec, minDM_vec)
r=length(WIS_M);%����������Ĳ���
I0=size(X_M,1)+1;%��0��������
N=size(X_M,2);%�ܵĲ���ʱ��
beta=0.01;%��һ������
VI_M{1}=[(X_M-minXM_vec*ones(1, N))./((maxXM_vec-minXM_vec)*ones(1,N))];%��һ������
if isempty(alpha0==1)
    alpha0=l;
end
for s=O:r-1
    VI_M{s+2}=BPDirectionCal(BPtype(s+1), WIS_M{s+1},VI_M{s+1},N,alpha0); %it#��1������
end
%����һ��
Y_M=(VI_M{r+1}-beta).*((maxDM_vec-minDM_vec)*ones( 1, N))/(1-2*beta)+minDM_vec*ones(1,N);

end