% Script demonstrating usage of the bpdn function.
%
% Author: Brendt Wohlberg <brendt@lanl.gov>  Modified: 2015-03-05
%
% This file is part of the SPORCO library. Details of the copyright
% and user license can be found in the 'Copyright' and 'License' files
% distributed with the library.


% Signal and dictionary size 信号和字典的大小
N = 512; 
M = 4*N;
% Number of non-zero coefficients in generator 非零系数的个数
L = 32;
% Noise level 噪声度
sigma = 0.5;

% Construct random dictionary and random sparse coefficients 构建随机字典和随机稀疏系数
D = randn(N, M); % 512x2048 产生正态分布的随机数
x0 = zeros(M, 1);% x0为稀疏系数
si = randperm(M);%M和字典的列数一致 从1-M的整数
si = si(1:L);%取了前32个
x0(si) = randn(L, 1);%将x0中随机的32个数替换成正态分布的随机数
% Construct reference and noisy signal %构建参考的噪声信号
s0 = D*x0;%代表的是第一个DxX构成的结果 是512x4的一维信号
s = s0 + sigma*randn(N,1);%加上噪声 

% BPDN for recovery of sparse representation 稀疏表示的恢复
lambda = 20;%lambda为20
opt = [];
opt.Verbose = 1;%是否显示迭代的标志
opt.rho = 100;%rho为100
opt.RelStopTol = 1e-6;%停止标志
%opt.AutoRho = 0;%进行测试，不适用autorho
%opt.AutoRhoScaling = 0;%进行测试，使用autorho但是不使用 对 tao的自适应。
size(D) %512 2048
size(s) %512 1
[x1, optinf] = bpdn(D, s, lambda, opt);

%后面是画图，先不考虑，但是这些打印图，结果的方法要学到
figure('position', [100, 100, 1200, 300]);
subplot(1,4,1);
plot(optinf.itstat(:,2));%打印的jfn，就是functional value
xlabel('Iterations');
ylabel('Functional value');

subplot(1,4,2);
semilogy(optinf.itstat(:,5));%打印的r，为primal残差
xlabel('Iterations');
ylabel('Primal residual');

subplot(1,4,3);
semilogy(optinf.itstat(:,6));%打印的s，为dual残差
xlabel('Iterations');
ylabel('Dual residual');

subplot(1,4,4);
semilogy(optinf.itstat(:,9));%打印的rho的变化
xlabel('Iterations');
ylabel('Penalty parameter');

%初始化稀疏系数和最后计算出来的系数化系数的对比。
figure;
plot(x0,'r');
hold on;
plot(x1,'b');
hold off;
legend('Reference', 'Recovered', 'Location', 'SouthEast');
title('Dictionary Coefficients');


%演示重新启动的功能，做40迭代和分别做20次+20次迭代
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
%只需要将这三个参数带进去就可以继续进行迭代，相当于初始化，字典不需要更新，D不变。
[x4, optinf4] = bpdn(D, s, lambda, opt4);


figure;
plot(optinf2.itstat(:,1), optinf2.itstat(:,2), 'r');
hold on;
plot(optinf3.itstat(:,1), optinf3.itstat(:,2), 'g');
plot(optinf4.itstat(:,1) + optinf3.itstat(end,1), optinf4.itstat(:,2), 'b');
xlabel('Iterations');
ylabel('Functional value');
legend('Uninterrupted', 'Stop at 20 iterations', 'Restart at 20 iterations');


% 构造稀疏系数和相应信号的集合。
% Construct ensemble of L-sparse coefficients and corresponding signals
K = 8;
X0 = zeros(M, K);%构造了一个稀疏矩阵的集合。
for l = 1:K,
  si = randperm(M);
  si = si(1:L);
  X0(si,l) = randn(L, 1);
end
S0 = D*X0;
S = S0 + sigma*randn(N, K);
% 恢复上述构建的稀疏系数矩阵
% BPDN for simultaneous recovery of sparse coefficient matrix
lambda = 20;
opt = [];
opt.Verbose = 1;
opt.rho = 100;
opt.RelStopTol = 1e-6;
[X1, optinf] = bpdn(D, S, lambda, opt);

%进行两个稀疏系数矩阵的对比。
figure;
subplot(1,2,1);
imagesc(X0);
title('Reference');
subplot(1,2,2);
imagesc(X1);
title('Recovered');
