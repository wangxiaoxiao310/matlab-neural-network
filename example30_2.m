function [WIS_M, En,maxXM_vec,minXM_vec, maxDM_vec,minDM_vec]=example30_2(X_M,D_M,BPtype,BPparameter, WIS_M, alpha0)
%�������˵��
%X_M���������룬��I0��1��*N���󣬸��б�ʾ��ά�������룬���б�ʾ��ʱ�������
%DM���������Ir*N���󣬸��б�ʾ��ά����������������б�ʾ��ʱ����������
%BPtype sigmoid������2Ϊ tansig����
%BPparameter��������Ĳ���������ΪILL��������Ľڵ�������IALL����2��6��4�ݱ�ʾ��һ���㣬��2���ڵ㣬�ڶ�������6���ڵ㣬����������4���ڵ�
%maxstep threshold��������ľ�����ֵ��etha0���������ѧϰ����
%mc�����Ӷ�������
%WIS_M����ʼ����Ȩ���󣬿�ѡ������������ʱ��
%alpha0 sigmoid�����Ĳ�����Ĭ��ֵΪ1
%�����ʽ�� BPparameteralpha��iall��threshold��etha0����
%�������˵��
%WIS_M:���������Ȩ����
%En��������ʱ�����
%max_XM vec min XM vec�� maxDM vec minDM vec���������롢������������Сֵ����������������������Сֵ����

%��һ����������ʼ��
[IALL, maxstep, threshold0,etha0,mc]=deal(BPparameter{:});%�õ�������ڵ����������������������ȡ�ѧϰ����(A=deal��B)�������б�B��ֵһһ��ֵ��A)
tic %��ʼ��ʱ
beta=0.01;%��һ����������һ����������beta��1��beta��
n=1;%������
I0=size(X_M,1);%��0�������������صڶ�ά��������������
[Ir,N]=size(D_M);%������Ĳ�������ʱ��,sizeδָ���ڶ���������ָ������ά����,�򶼷���,NΪ��������
if size(IALL,2)>1
    IALL=IALL';%����Ǻ����������Ϊ������
end

IALL=[I0;IALL;Ir];
%��һ����������
minXM_vec= min(X_M,[],2);maxXM_vec=max(X_M,[],2);%�������������ά���������Сֵ
minDM_vec= min(D_M,[],2);maxDM_vec=max(D_M,[],2);%�������������ά���������Сֵ
VI_M{1}=[(X_M-minXM_vec*ones(1, N))./((maxXM_vec-minXM_vec)*ones(1,N))];%��һ������
H_M{1}=(D_M-minDM_vec*ones(1, N))./((maxDM_vec-minDM_vec)*ones(1,N))*(1-2*beta)+beta;%��һ�����
%�����������
r=length(IALL)-1;
%����BP�����缤����Ƿ�ÿ�㶼��
if length(BPtype)==1   %���ֻ����һ����������Ϊÿ�㶼����������
     BPtype=ones(1,r);
else
    if length(BPtype)~=r
        error(['������ĸ���Ϊ',num2str(length(BPtype)),'�������������������Ϊ',num2str(r),'�������߲�ƥ��.']);
    end
end
numP=0;%�����������д���Ȩֵ����
 for s=1: r
    numP=numP+(IALL(s)+1)*IALL(s+1);
 end
disp(['BP�����繲��',num2str(numP),'�������Ȩֵ��',num2str(10),'x',num2str(N),'ά����:',num2str(Ir), 'x',num2str(N),'ά���.'])
disp(['BP������I1��',num2str(r),'��ļ���������'])
disp(['�� ',num2str(find(BPtype==1)),'��Ϊ sigmoid���������',num2str(find(BPtype==2)),'��Ϊ tansig�����'])
%��¼�����
REA_vec=[];
if isempty(alpha0)==1 %��� alpha0����Ϊ�ռ���������Ϊ1
     alpha0=l;
end
if isempty(WIS_M)==1 %����������ʼ���Ӿ�����������ɳ�ʼ���Ӿ���
    for s=0:r-1
        WIS_M{s+1}=normand(0,1,IALL(s+1),IALL(s+2));%�����s��1�������Ȩ����
        VI_M{s+2}=BPDirectionCal(BPtype(s+1), WIS_M{s+1},VI_M{s+1},N,alpha0); %���� s+1������
     end
else
    for s=0:r-1
        VI_M{s+2}=BPDirectionCal(BPtype(s+1), WIS_M{s+1},VI_M{s+1},N,alpha0); %���� s+1������

    end
end
DeltaWIS_M=WIS_M;%���ʼȨֵ�����������Ȩֵ����
%���������
EA_M=H_M-VI_M{r+1};
En=norm(EA_M, 'fro')/sqrt(2);%����f-����������ֵƽ���ͣ�
%�ڶ��������ñ�������Ķ��������������������������Ȩֵ�������
while En>threshold0 && n<maxstep
    GammaIr_M{1}=DerivatBP(BPtype(r),VI_M{r+1},alpha0).*EA_M';%���������
    DeltaWIS_M{r}=mc*DeltaWIS_M{r}+(1-mc)*etha0*[VI_M{r}; ones(1, N)]*GammaIr_M{1};%���ø��Ӷ���������Ȩֵ����
    WIS_M{r}=WIS_M{s}+DeltaWIS_M{r};%���������Ȩ���������
    % BP���������ķ��򴫲�
    for s=r-1:-1: 1
         %����s������
         GammaIr_M{r+l-s}=DerivatBP(BPtype(s),VI_M{s+1},alpha0).*(GammaIr_M{r-s}*WIS_M{s+1}(1:ALL(s+1),:)');
         %���ø��Ӷ���������Ȩֵ����
         DeltaWIS_M{s}=mc*DeltaWIS_M{s}+(1-mc)*etha0*[VI_M{s};ones(1,N)]*GammaIr_M{r+1-s};
         %����Is������Ȩ���������
         WIS_M{s}=WIS_M{s}+DeltaWIS_M{s};
    end
    n=n+1;
    %��������1
    %�����������ʵ�����
    for s=0: r-1
        VI_M{s+2}=BPDirectionCal(BPtype(s+1), WIS_M{s+1},VI_M{s+1},N,alpha0); %���� s+1������
    end
    %���������
    EA_M=H_M-VI_M{r+1};
    En=norm(EA_M,'fro')/sqrt(2);
    REA_vec=[REA_vec;En];
 end
%�����������沢��ʾ���
%��ѵ���õĽ�����浽�ļ�BPmatrixmat��
save('BPmatrix.mat',' WIS_M', 'maxXM_vec','minXM_vec',' maxDMvec','minDM_vec')
t=toc;%������ʱ
t1=1:n-1;%��Ϊn�����һ�Σ�����Ҫ��1
figure ('name',['BP������ķ���ʱ��Ϊ',num2str(t0),'��'],'numbertitle','off')
plot(tl, REA_vec,' linewidth', 1.5);
grid on;
xlabel('��������',' fontsize',16);
ylabel('���',' fontsize',16);
title('��ƫ�õ�BP�������������������仯�����',' fontsize',16);
set(gca,'fontsize', 16); 
%����һ��
Y_M=(VI_M{r+1}-beta).*((maxDM_vec-minDM_vec)*ones(1, N))./(1-2*beta)+minDM_vec*ones(1,N);
t2=1:N;
%��ͼ
for s0=1: ceil(Ir/2)
    if s0<ceil(Ir/2)||mod(lr,2)==0
        figure('name',' BP������������� ','numbertitle','off')
        for sl=1: 2
            subplot(2,1,sl);
            plot(t2,D_M(2*(s0-1)+s1, :),t2,Y_M(2*(s0-1)+s1,:) ,'linewidth', 1.5);
            grid on;
            legend('�������','BP���������',16)
            xlabel('����ʱ��',' fontsize',16);
            ylabel(['��',num2str(2*(s0-1)+s1),'ά���'],' fontsize',16);
            title('���������BP����������Ƚ�',' fontsize',16);
            set(gca,'fontsize', 16); 
        end
    else
      figure('name',' BP������������� ','numbertitle','off')
      plot(t2,D_M(Ir,:),t2,Y_M(Ir,:) ,'linewidth', 1.5);
      grid on;
     legend('�������','BP���������',16)
     xlabel('����ʱ��',' fontsize',16);
     ylabel(['��',num2str(Ir),'ά���'],' fontsize',16);
     title('���������BP����������Ƚ�',' fontsize',16);
      set(gca,'fontsize', 16); 
   end
end





end

