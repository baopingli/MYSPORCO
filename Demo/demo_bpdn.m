% Script demonstrating usage of the bpdn function.
%
% Author: Brendt Wohlberg <brendt@lanl.gov>  Modified: 2015-03-05
%
% This file is part of the SPORCO library. Details of the copyright
% and user license can be found in the 'Copyright' and 'License' files
% distributed with the library.


% Signal and dictionary size �źź��ֵ�Ĵ�С
N = 512; 
M = 4*N;
% Number of non-zero coefficients in generator ����ϵ���ĸ���
L = 32;
% Noise level ������
sigma = 0.5;

% Construct random dictionary and random sparse coefficients ��������ֵ�����ϡ��ϵ��
D = randn(N, M); % 512x2048 ������̬�ֲ��������
x0 = zeros(M, 1);% x0Ϊϡ��ϵ��
si = randperm(M);%M���ֵ������һ�� ��1-M������
si = si(1:L);%ȡ��ǰ32��
x0(si) = randn(L, 1);%��x0�������32�����滻����̬�ֲ��������
% Construct reference and noisy signal %�����ο��������ź�
s0 = D*x0;%������ǵ�һ��DxX���ɵĽ�� ��512x4��һά�ź�
s = s0 + sigma*randn(N,1);%�������� 

% BPDN for recovery of sparse representation ϡ���ʾ�Ļָ�
lambda = 20;%lambdaΪ20
opt = [];
opt.Verbose = 1;%�Ƿ���ʾ�����ı�־
opt.rho = 100;%rhoΪ100
opt.RelStopTol = 1e-6;%ֹͣ��־
%opt.AutoRho = 0;%���в��ԣ�������autorho
%opt.AutoRhoScaling = 0;%���в��ԣ�ʹ��autorho���ǲ�ʹ�� �� tao������Ӧ��
size(D) %512 2048
size(s) %512 1
[x1, optinf] = bpdn(D, s, lambda, opt);

%�����ǻ�ͼ���Ȳ����ǣ�������Щ��ӡͼ������ķ���Ҫѧ��
figure('position', [100, 100, 1200, 300]);
subplot(1,4,1);
plot(optinf.itstat(:,2));%��ӡ��jfn������functional value
xlabel('Iterations');
ylabel('Functional value');

subplot(1,4,2);
semilogy(optinf.itstat(:,5));%��ӡ��r��Ϊprimal�в�
xlabel('Iterations');
ylabel('Primal residual');

subplot(1,4,3);
semilogy(optinf.itstat(:,6));%��ӡ��s��Ϊdual�в�
xlabel('Iterations');
ylabel('Dual residual');

subplot(1,4,4);
semilogy(optinf.itstat(:,9));%��ӡ��rho�ı仯
xlabel('Iterations');
ylabel('Penalty parameter');

%��ʼ��ϡ��ϵ���������������ϵ����ϵ���ĶԱȡ�
figure;
plot(x0,'r');
hold on;
plot(x1,'b');
hold off;
legend('Reference', 'Recovered', 'Location', 'SouthEast');
title('Dictionary Coefficients');


%��ʾ���������Ĺ��ܣ���40�����ͷֱ���20��+20�ε���
% Illustrate restart capability: do 40 iterations and compare with
% 20 iterations and then restart for an additional 20 iterations
opt2 = opt;
opt2.MaxMainIter = 40;
[x2, optinf2] = bpdn(D, s, lambda, opt2);

opt3 = opt2;
opt3.MaxMainIter = 20;
[x3, optinf3] = bpdn(D, s, lambda, opt3);

opt4 = opt3;
opt4.Y0 = optinf3.Y;
opt4.U0 = optinf3.U;
opt4.rho = optinf3.rho;
%ֻ��Ҫ����������������ȥ�Ϳ��Լ������е������൱�ڳ�ʼ�����ֵ䲻��Ҫ���£�D���䡣
[x4, optinf4] = bpdn(D, s, lambda, opt4);


figure;
plot(optinf2.itstat(:,1), optinf2.itstat(:,2), 'r');
hold on;
plot(optinf3.itstat(:,1), optinf3.itstat(:,2), 'g');
plot(optinf4.itstat(:,1) + optinf3.itstat(end,1), optinf4.itstat(:,2), 'b');
xlabel('Iterations');
ylabel('Functional value');
legend('Uninterrupted', 'Stop at 20 iterations', 'Restart at 20 iterations');


% ����ϡ��ϵ������Ӧ�źŵļ��ϡ�
% Construct ensemble of L-sparse coefficients and corresponding signals
K = 8;
X0 = zeros(M, K);%������һ��ϡ�����ļ��ϡ�
for l = 1:K,
  si = randperm(M);
  si = si(1:L);
  X0(si,l) = randn(L, 1);
end
S0 = D*X0;
S = S0 + sigma*randn(N, K);
% �ָ�����������ϡ��ϵ������
% BPDN for simultaneous recovery of sparse coefficient matrix
lambda = 20;
opt = [];
opt.Verbose = 1;
opt.rho = 100;
opt.RelStopTol = 1e-6;
[X1, optinf] = bpdn(D, S, lambda, opt);

%��������ϡ��ϵ������ĶԱȡ�
figure;
subplot(1,2,1);
imagesc(X0);
title('Reference');
subplot(1,2,2);
imagesc(X1);
title('Recovered');
