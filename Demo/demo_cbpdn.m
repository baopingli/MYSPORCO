% Script demonstrating usage of the cbpdn function.
%
% Author: Brendt Wohlberg <brendt@lanl.gov>  Modified: 2015-04-09
%
% This file is part of the SPORCO library. Details of the copyright
% and user license can be found in the 'Copyright' and 'License' files
% distributed with the library.

% 加载字典
% Load dictionary
%sporco_path='D:\matlab_sporco\';
%load([sporco_path '/Data/ConvDict.mat']);
load(['D:/matlab_sporco/Data/ConvDict.mat']);
%key -value映射的一种类型创建的一种map容器
dmap = containers.Map(ConvDict.Label, ConvDict.Dict);
D = dmap('12x12x36');%D是filters，filters固定了，去优化Z D的大小为12 12 36

% 导入图片进行灰度的处理
% Load test image
%s = single(rgbtogrey(stdimage('lena')))/255;
s = single(rgbtogrey(stdimage('monarch')))/255;
%s=imread('./Data/Std/lena.grey.png');
if isempty(s),
  error('Data required for demo scripts has not been installed.');
end

% Highpass filter test image
%将测试图像高通滤波
npd = 16;
fltlmbd = 5;
[sl, sh] = lowpass(s, fltlmbd, npd);%后面使用的是高通滤波的部分
%sl 低通 sh 高通
% Compute representation
lambda = 0.01;
opt = [];
opt.Verbose = 1;%显示详细结果
opt.MaxMainIter = 500;%最大迭代次数为500次
opt.rho = 100*lambda + 1;%初始化rho为2
opt.RelStopTol = 1e-3;%相对收敛公差
opt.AuxVarObj = 1;%是否计算目标函数
opt.HighMemSolve = 1;%使用更多的内存获取更快的解决
[X, optinf] = cbpdn(D, sh, lambda, opt);

% Compute reconstruction
DX = ifft2(sum(bsxfun(@times, fft2(D, size(X,1), size(X,2)), fft2(X)),3), ...
           'symmetric');
%bsxfun(fun,A,B)对两个矩阵A、B的每一个元素进行指定的计算。

figure;
subplot(1,3,1);
plot(optinf.itstat(:,2));
xlabel('Iterations');
ylabel('Functional value');
subplot(1,3,2);
semilogy(optinf.itstat(:,5));
xlabel('Iterations');
ylabel('Primal residual');
subplot(1,3,3);
semilogy(optinf.itstat(:,6));
xlabel('Iterations');
ylabel('Dual residual');


figure;
imagesc(sum(abs(X),3));
title('Sum of absolute value of coefficient maps');


figure;
subplot(1,3,1);
imdisp(s);
title('Original image');
subplot(1,3,2);
imdisp(DX + sl);
title(sprintf('Reconstructed image (SNR: %.2fdB)', snr(s, DX + sl)));
subplot(1,3,3);
imagesc(DX + sl - s);
axis image; axis off;
title('Difference');
