function [Y, optinf] = cbpdn(D, S, lambda, opt)

% cbpdn -- Convolutional Basis Pursuit DeNoising
%
%         argmin_{x_m} (1/2)||\sum_m d_m * x_m - s||_2^2 +
%                           lambda \sum_m ||x_m||_1
%
%         The solution is computed using an ADMM approach (see
%         boyd-2010-distributed) with efficient solution of the main
%         linear systems (see wohlberg-2016-efficient).
%
% Usage:
%       [Y, optinf] = cbpdn(D, S, lambda, opt);
%
% Input:
%       D           Dictionary filter set (3D array)�ֵ�
%       S           Input imageͼ��
%       lambda      Regularization parameter����ϵ��
%       opt         Algorithm parameters structure�Ż�������
%
% Output:
%       Y           Dictionary coefficient map set (3D array)�������ֵ��ϡ��ϵ��
%       optinf      Details of optimisation �Ż���ϸ����
%
%
% Options structure fields:
%   Verbose          Flag determining whether iteration status is displayed.
%                    Fields are iteration number, functional value,
%                    data fidelity term, l1 regularisation term, and
%                    primal and dual residuals (see Sec. 3.3 of
%                    boyd-2010-distributed). The value of rho is also
%                    displayed if options request that it is automatically
%                    adjusted.�Ƿ���Ϣȫ����ʾ
%   MaxMainIter      Maximum main iterations�����������
%   AbsStopTol       Absolute convergence tolerance (see Sec. 3.3.1 of
%                    boyd-2010-distributed)������������
%   RelStopTol       Relative convergence tolerance (see Sec. 3.3.1 of
%                    boyd-2010-distributed)�����������
%   L1Weight         Weighting array for coefficients in l1 norm of X
%   һ��ʽ�еļ�Ȩ����
%   Y0               Initial value for Y ��ʼ��ϵ�����֣� Yһ�����Ǹ��Թ������ƽ�X���Ǹ�
%   U0               Initial value for U ��ʼ��U
%   rho              ADMM penalty parameter 
%   AutoRho          Flag determining whether rho is automatically updated
%                    (see Sec. 3.4.1 of boyd-2010-distributed)
%   AutoRhoPeriod    Iteration period on which rho is updated
%   RhoRsdlRatio     Primal/dual residual ratio in rho update test
%   RhoScaling       Multiplier applied to rho when updated
%   AutoRhoScaling   Flag determining whether RhoScaling value is
%                    adaptively determined (see wohlberg-2015-adaptive). If
%                    enabled, RhoScaling specifies a maximum allowed
%                    multiplier instead of a fixed multiplier.
%   RhoRsdlTarget    Residual ratio targeted by auto rho update policy.
%   StdResiduals     Flag determining whether standard residual definitions
%                    (see Sec 3.3 of boyd-2010-distributed) are used instead
%                    of normalised residuals (see wohlberg-2015-adaptive)
%   RelaxParam       Relaxation parameter (see Sec. 3.4.3 of
%                    boyd-2010-distributed)
%   NonNegCoef       Flag indicating whether solution should be forced to
%                    be non-negative
%   NoBndryCross     Flag indicating whether all solution coefficients
%                    corresponding to filters crossing the image boundary
%                    should be forced to zero.
%   AuxVarObj        Flag determining whether objective function is computed
%                    using the auxiliary (split) variable
%   HighMemSolve     Use more memory for a slightly faster solution
%
%
% Author: Brendt Wohlberg <brendt@lanl.gov>  Modified: 2015-12-28
%
% This file is part of the SPORCO library. Details of the copyright
% and user license can be found in the 'License' file distributed with
% the library.


if nargin < 4,
  opt = [];
end
checkopt(opt, defaultopts([]));
%�ָ�һ��Ĭ�ϵĲ���
opt = defaultopts(opt);

% Set up status display for verbose operation
hstr = 'Itn   Fnc       DFid      l1        r         s      ';
sfms = '%4d %9.2e %9.2e %9.2e %9.2e %9.2e';
nsep = 54;
%Ĭ������Ӧrho
if opt.AutoRho,
  hstr = [hstr '   rho  '];
  sfms = [sfms ' %9.2e'];
  nsep = nsep + 10;
end
if opt.Verbose && opt.MaxMainIter > 0,
  disp(hstr);
  disp(char('-' * ones(1,nsep)));
end

% Collapsing of trailing singleton dimensions greatly complicates
% handling of both SMV and MMV cases. The simplest approach would be
% if S could always be reshaped to 4d, with dimensions consisting of
% image rows, image cols, a single dimensional placeholder for number
% of filters, and number of measurements, but in the single
% measurement case the third dimension is collapsed so that the array
% is only 3d.
%size(sh)=768 512  size(sh,3)=size(sh,4)=1  3Ϊ1��ʾ�ǻҶ�ͼ��4Ϊ1��ʾһ��ͼƬ
%��ʵͼ���size��sh,3�����ص�Ӧ����ͼƬ��������size��D��3�����˲����ĸ�����xsz������
%�˲����Ĵ�СΪ12x12������Ϊ36����ϡ��ϵ���Ĵ�СӦ��Ϊ512x768x36��
if size(S,3) > 1 && size(S,4) == 1,
    %xsz��ϡ��ϵ��Ӧ���еĽṹ��Ϊ512 768 36 1�Ĵ�С �� �� �˲������� ͼƬ����
  xsz = [size(S,1) size(S,2) size(D,3) size(S,3)];
  % Insert singleton 3rd dimension (for number of filters) so that
  % 4th dimension is number of images in input s volume
  %��ͼ����512 768 1 ͼƬ������  ��ʽreshape
  S = reshape(S, [size(S,1) size(S,2) 1 size(S,3)]);
else
    %ִ�е����������
  xsz = [size(S,1) size(S,2) size(D,3) size(S,4)];
  %xsz�Ĵ�СΪ512 768 36 1 �������Ϊϡ��ϡ��Ĵ�С��
  %fprintf("123");
end

% Start timer
tstart = tic;

% Compute filters in DFT domain
% �������ָ�ʽ���ж�D���и���Ҷ�任
Df = fft2(D, size(S,1), size(S,2));%512 768 36
size(Df)
%����s��512 768�Ĵ�С����D���и���Ҷ�任
% Convolve-sum and its Hermitian transpose
%����Ͷ���ת��
Dop = @(x) sum(bsxfun(@times, Df, x), 3);
%��x��DF�ľ�����,sum��������ͨ����ɫ�ؽ������ x * DF���Ƚ��и���Ҷ���µ����Ȼ������ӣ�����Ǻ�Df�������
DHop = @(x) bsxfun(@times, conj(Df), x);%��x��DF�ĸ����������x * ��DF��T�������ֱ�ӽ������û�н�����ӡ�
%���㸵��Ҷ���µ�DS
% Compute signal in DFT domain
Sf = fft2(S);%����ҶS
size(Sf)
% S convolved with all filters in DFT domain
DSf = DHop(Sf);%DSf=fft2��S��*(DF)T ����Ƶ���µ� S * Dt
size(DSf)

% Default lambda is 1/10 times the lambda value beyond which the
% solution is a zero vector
if nargin < 3 | isempty(lambda),
  b = ifft2(DHop(Sf), 'symmetric');%���任����ͼ��s�任��ʱ���� ��DS���任����ʼ������lambda
  lambda = 0.1*max(vec(abs(b)));%���lambdaΪ�յĻ�����ʼ��lambda
end

% Set up algorithm parameters and initialise variables
rho = opt.rho;%���ò���rho
%���rhoΪ�ճ�ʼ��rho
if isempty(rho), rho = 50*lambda+1; end;
%����в�ȵ�ָ��Ϊ������Ϊ1
if isempty(opt.RhoRsdlTarget),
    %�����׼�в�Ϊ��
  if opt.StdResiduals,
    opt.RhoRsdlTarget = 1;
    %������ݱ�׼�в�����ʼ���в�ȵ�ָ��
  else
    opt.RhoRsdlTarget = 1 + (18.3).^(log10(lambda) + 1);
  end
end
%���ʹ�ø�����ڴ�����ȡ���ٵĽ��
%��������ǰ������C=D/��D*Dt+rho����Ȼ�󱣴棬ռ�����ڴ浫�ǿ��Կ�����⡣
if opt.HighMemSolve,
  C = bsxfun(@rdivide, Df, sum(Df.*conj(Df), 3) + rho);
else
  C = [];
end
Nx = prod(xsz);%����ϵ��ÿһ�еĳ˻�
optinf = struct('itstat', [], 'opt', opt);%�����ṹ��
r = Inf;%����primal�в�
s = Inf;%����dual�в�
epri = 0;%����ֹͣ������
edua = 0;

% Initialise main working variables
X = [];
if isempty(opt.Y0),
  Y = zeros(xsz, class(S));%��Y��ʼ����Xһ����С
else
  Y = opt.Y0;
end
Yprv = Y;
if isempty(opt.U0),
  if isempty(opt.Y0),
    U = zeros(xsz, class(S));
    %����S������ȥ��ʼ��U
  else
    U = (lambda/rho)*sign(Y);
  end
else
  U = opt.U0;
end

% Main loop
k = 1;
while k <= opt.MaxMainIter && (r > epri | s > edua),

  % Solve X subproblem
  Xf = solvedbi_sm(Df, rho, DSf + rho*fft2(Y - U), C);
  X = ifft2(Xf, 'symmetric');

  % See pg. 21 of boyd-2010-distributed
  if opt.RelaxParam == 1,
    Xr = X;
  else
    Xr = opt.RelaxParam*X + (1-opt.RelaxParam)*Y;
  end

  % Solve Y subproblem
  Y = shrink(Xr + U, (lambda/rho)*opt.L1Weight);
  if opt.NonNegCoef,
    Y(Y < 0) = 0;
  end
  if opt.NoBndryCross,
    Y((end-size(D,1)+2):end,:,:,:) = 0;
    Y(:,(end-size(D,2)+2):end,:,:) = 0;
  end

  % Update dual variable
  U = U + Xr - Y;

  % Compute data fidelity term in Fourier domain (note normalisation)
  if opt.AuxVarObj,
    Yf = fft2(Y); % This represents unnecessary computational cost
    Jdf = sum(vec(abs(sum(bsxfun(@times,Df,Yf),3)-Sf).^2))/(2*xsz(1)*xsz(2));
    Jl1 = sum(abs(vec(bsxfun(@times, opt.L1Weight, Y))));
  else
    Jdf = sum(vec(abs(sum(bsxfun(@times,Df,Xf),3)-Sf).^2))/(2*xsz(1)*xsz(2));
    Jl1 = sum(abs(vec(bsxfun(@times, opt.L1Weight, X))));
  end
  Jfn = Jdf + lambda*Jl1;

  nX = norm(X(:)); nY = norm(Y(:)); nU = norm(U(:));
  if opt.StdResiduals,
    % See pp. 19-20 of boyd-2010-distributed
    r = norm(vec(X - Y));
    s = norm(vec(rho*(Yprv - Y)));
    epri = sqrt(Nx)*opt.AbsStopTol+max(nX,nY)*opt.RelStopTol;
    edua = sqrt(Nx)*opt.AbsStopTol+rho*nU*opt.RelStopTol;
  else
    % See wohlberg-2015-adaptive
    r = norm(vec(X - Y))/max(nX,nY);
    s = norm(vec(Yprv - Y))/nU;
    epri = sqrt(Nx)*opt.AbsStopTol/max(nX,nY)+opt.RelStopTol;
    edua = sqrt(Nx)*opt.AbsStopTol/(rho*nU)+opt.RelStopTol;
  end

  % Record and display iteration details
  tk = toc(tstart);
  optinf.itstat = [optinf.itstat; [k Jfn Jdf Jl1 r s epri edua rho tk]];
  if opt.Verbose,
    if opt.AutoRho,
      disp(sprintf(sfms, k, Jfn, Jdf, Jl1, r, s, rho));
    else
      disp(sprintf(sfms, k, Jfn, Jdf, Jl1, r, s));
    end
  end

  % See wohlberg-2015-adaptive and pp. 20-21 of boyd-2010-distributed
  if opt.AutoRho,
    if k ~= 1 && mod(k, opt.AutoRhoPeriod) == 0,
      if opt.AutoRhoScaling,
        rhomlt = sqrt(r/(s*opt.RhoRsdlTarget));
        if rhomlt < 1, rhomlt = 1/rhomlt; end
        if rhomlt > opt.RhoScaling, rhomlt = opt.RhoScaling; end
      else
        rhomlt = opt.RhoScaling;
      end
      rsf = 1;
      if r > opt.RhoRsdlTarget*opt.RhoRsdlRatio*s, rsf = rhomlt; end
      if s > (opt.RhoRsdlRatio/opt.RhoRsdlTarget)*r, rsf = 1/rhomlt; end
      rho = rsf*rho;
      U = U/rsf;
      fprintf('rhorsdltarget:%5.5g\n',opt.RhoRsdlTarget);
      if opt.HighMemSolve && rsf ~= 1,
        C = bsxfun(@rdivide, Df, sum(Df.*conj(Df), 3) + rho);
      end
    end
  end

  Yprv = Y;
  k = k + 1;

end

% Record run time and working variables
optinf.runtime = toc(tstart);
optinf.X = X;
optinf.Xf = Xf;
optinf.Y = Y;
optinf.U = U;
optinf.lambda = lambda;
optinf.rho = rho;

% End status display for verbose operation
if opt.Verbose && opt.MaxMainIter > 0,
  disp(char('-' * ones(1,nsep)));
end

return


function u = vec(v)

  u = v(:);

return


function u = shrink(v, lambda)

  if isscalar(lambda),
    u = sign(v).*max(0, abs(v) - lambda);
  else
    u = sign(v).*max(0, bsxfun(@minus, abs(v), lambda));
  end

return


function opt = defaultopts(opt)

  if ~isfield(opt,'Verbose'),
    opt.Verbose = 0;
  end
  if ~isfield(opt,'MaxMainIter'),
    opt.MaxMainIter = 1000;
  end
  if ~isfield(opt,'AbsStopTol'),
    opt.AbsStopTol = 0;
  end
  if ~isfield(opt,'RelStopTol'),
    opt.RelStopTol = 1e-4;
  end
  if ~isfield(opt,'L1Weight'),
    opt.L1Weight = 1;
  end
  if ~isfield(opt,'Y0'),
    opt.Y0 = [];
  end
  if ~isfield(opt,'U0'),
    opt.U0 = [];
  end
  if ~isfield(opt,'rho'),
    opt.rho = [];
  end
  if ~isfield(opt,'AutoRho'),
    opt.AutoRho = 1;
  end
  if ~isfield(opt,'AutoRhoPeriod'),
    opt.AutoRhoPeriod = 1;
  end
  if ~isfield(opt,'RhoRsdlRatio'),
    opt.RhoRsdlRatio = 1.2;
  end
  if ~isfield(opt,'RhoScaling'),
    opt.RhoScaling = 100;
  end
  if ~isfield(opt,'AutoRhoScaling'),
    opt.AutoRhoScaling = 1;
  end
  if ~isfield(opt,'RhoRsdlTarget'),
    opt.RhoRsdlTarget = [];
  end
  if ~isfield(opt,'StdResiduals'),
    opt.StdResiduals = 0;
  end
  if ~isfield(opt,'RelaxParam'),
    opt.RelaxParam = 1.8;
  end
  if ~isfield(opt,'NonNegCoef'),
    opt.NonNegCoef = 0;
  end
  if ~isfield(opt,'NoBndryCross'),
    opt.NoBndryCross = 0;
  end
  if ~isfield(opt,'AuxVarObj'),
    opt.AuxVarObj = 0;
  end
  if ~isfield(opt,'HighMemSolve'),
    opt.HighMemSolve = 0;
  end

return
