% Script demonstrating usage of the cbpdn function.
%
% Author: Brendt Wohlberg <brendt@lanl.gov>  Modified: 2015-04-09
%
% This file is part of the SPORCO library. Details of the copyright
% and user license can be found in the 'Copyright' and 'License' files
% distributed with the library.

% �����ֵ�
% Load dictionary
%sporco_path='D:\matlab_sporco\';
%load([sporco_path '/Data/ConvDict.mat']);
load(['D:/matlab_sporco/Data/ConvDict.mat']);
%key -valueӳ���һ�����ʹ�����һ��map����
dmap = containers.Map(ConvDict.Label, ConvDict.Dict);
D = dmap('12x12x36');%D��filters��filters�̶��ˣ�ȥ�Ż�Z D�Ĵ�СΪ12 12 36

% ����ͼƬ���лҶȵĴ���
% Load test image
%s = single(rgbtogrey(stdimage('lena')))/255;
s = single(rgbtogrey(stdimage('monarch')))/255;
%s=imread('./Data/Std/lena.grey.png');
if isempty(s),
  error('Data required for demo scripts has not been installed.');
end

% Highpass filter test image
%������ͼ���ͨ�˲�
npd = 16;
fltlmbd = 5;
[sl, sh] = lowpass(s, fltlmbd, npd);%����ʹ�õ��Ǹ�ͨ�˲��Ĳ���
%sl ��ͨ sh ��ͨ
% Compute representation
lambda = 0.01;
opt = [];
opt.Verbose = 1;%��ʾ��ϸ���
opt.MaxMainIter = 500;%����������Ϊ500��
opt.rho = 100*lambda + 1;%��ʼ��rhoΪ2
opt.RelStopTol = 1e-3;%�����������
opt.AuxVarObj = 1;%�Ƿ����Ŀ�꺯��
opt.HighMemSolve = 1;%ʹ�ø�����ڴ��ȡ����Ľ��
[X, optinf] = cbpdn(D, sh, lambda, opt);

% Compute reconstruction
DX = ifft2(sum(bsxfun(@times, fft2(D, size(X,1), size(X,2)), fft2(X)),3), ...
           'symmetric');
%bsxfun(fun,A,B)����������A��B��ÿһ��Ԫ�ؽ���ָ���ļ��㡣

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
