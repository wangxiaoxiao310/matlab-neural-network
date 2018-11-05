close all;clc
%����200������������
N=200;
Ntrain=150;
x1_vec=linspace(-2,2,N)'; %������-2��2Ϊʼĩ��N��Ԫ�ص�������
x2_vec=unifrnd(-2,2,N,1); %���ɣ����������ȷֲ��������:����һ��N*1���飬������Ԫ��Ϊ-2~2�������
x3_vec=x1_vec;  
x4_vec=unifrnd(-2,2,N,1);
X_Data=[x1_vec, x2_vec, x3_vec,x4_vec]';%������������ʽ
X0_M=X_Data(:,1:Ntrain);
X0_test=X_Data(:,Ntrain+1:N);
%�õ��������
d1_vec=(x1_vec+x2_vec.^2+x3_vec.^3+x4_vec.^4)./(x1_vec.^2+x2_vec.^2+1);
d2_vec=(exp(-abs(x1_vec+x2_vec))+2)./(x3_vec.^2+x4_vec.^2+1);
%�������ά��ͳ��
D_Data=[d1_vec,d2_vec]';%����ֵ�ľ�����ʽ�������ʱ��ͳ��
D0_M=D_Data(:,1:Ntrain);
D0_test=D_Data(:,Ntrain+1:N);

%Step2����ʼ�������������
IALL=[9,9,9]; %��11����Ir��1���������ڵ���
BPtype=[1,2,1,2]; %��11��r��������紫�ݺ�����1Ϊ sigmoid tansig����,2Ϊtansig����
r=length(BPtype);%�õ�������Ĳ���
maxstep=40000; %���������ѵ������
threshold0=1e-2;%������ľ�����ֵ
etha0=1e-3;%�������ѧϰ����
mc=0;%���Ӷ�������
%��������ڵ��������ѵ��������������ֵ��ѧϰ���ʡ�������������Ϊһ������
BPparameter={IALL, maxstep,threshold0,etha0,mc};
WIS_M=[];%��ʼ����Ȩ����,��ѡ����,������ʱ��[]
alpha0=1;%sigmoid �����Ĳ�����Ĭ��ֵΪ1

[WIS_M, En,maxXM_vec,minXM_vec, maxDM_vec,minDM_vec]=example30_2(X0_M,D0_M,BPtype,BPparameter, WIS_M, alpha0);
%Step3�����ö�������纯�� example30��2������ز������������ʾ����������������ͼ30��2��ͼ30��3��ʾ��
%�����纯������Ĳ���Ϊ
%WISM�����������Ȩ����
%En����������ʱ�����
%maxXM_vec,minXM_vec, maxDM_vec,minDM_vec����������������������Сֵ����������������������Сֵ����
 
disp('��������õ����������Ȩ��������')
for i=1: 4
    disp(['��I',num2str(i-1),'������num2str��i���������Ȩ����Ϊ��']);
    WIS_M{i}
 end
disp(['����������Ϊ',num2str(En)]);

%step4:���ò�����������������в���
Y_M=example30_3(x0_test, BPtype, alpha0, WIS_M, maxXM_vec, minXM_vec, maxDM_vec, minDM_vec);
SampleNum=(Ntrain+1:N);%������ż�
figure('name','����������������ʵ������ıȽ�','numbertitle',off);
subplot(211)
plot(SampleNum, D0_test(1,: ), SampleNum,Y_M(1,: ),'linewidth',3);
grid on:
h=legend('�������','ʵ�����');
set(h,'fontsize',16);
title('���ò��������õ�������ĵ�һά���������ʵ������ıȽ�','fontsize',16);
xlabel('�������',' fontsize',16);
ylabel('���',' fontsize',16);
set(gca,'fontsize', 16); 
subplot(212)
plot(SampleNum, D0_test(2,: ), SampleNum,Y_M(2,: ),'linewidth',3);
grid on:
h=legend('�������','ʵ�����'); set(h,fontsize, 16);
set(h,'fontsize',16);
title('���ò��������õ�������ĵڶ�ά���������ʵ������ıȽ�','fontsize',16);
xlabel('�������',' fontsize',16);
ylabel('���',' fontsize',16);
set(gca,'fontsize', 16); subplot(212)

