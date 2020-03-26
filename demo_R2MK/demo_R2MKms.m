function demo_R2MKms
n = 1;
[img, img_gt, nClass, rows, cols, bands] = load_datas(n);

switch n,
    case 1, type = 2; rmk_sig = 0.25; rmk_lam = 1e-6;
        eta = 0.5; tau = 1e-5;
        nCs = [20 50 100 200 400 800];
end
sigs = [0.02 : 0.02 : 0.1];

rmk_time = zeros(2,1);
t_begin = tic;
labels = SuperPixelMultiScale(img, rows, cols, nCs, sigs);
rmk_time(1) = toc(t_begin);


nt = 3; it = 1;
[train_idx, test_idx] = load_train_test(n, type, nt, it);
[Train, Test] = set_train_test(train_idx, test_idx, img, img_gt);

t_begin = tic;
P0 = RBF(img(:,Train.idx), img, rmk_sig, 2000);
rmk_pred = RegionRelaxedMultipleKernel(img, Train.idx, labels, rmk_sig, ...
    rmk_lam, tau, eta, P0, Train.lab);
rmk_time(2) = toc(t_begin);
rmk_acc = class_eval(rmk_pred(Test.idx), Test.lab);
disp(rmk_acc);
end

function pred = RegionRelaxedMultipleKernel(img, trainidx, labels, sig, ...
    lam, tau, eta, P0, lab)
train_size = length(trainidx);
labels_size = size(labels,2); eta_labels_size = eta / labels_size;
P = (1-eta) * P0;
G = P(:,trainidx) * P(:,trainidx)';
for i = 1 : labels_size,
    label = labels(:, i);
    im = [SuperpixelMean(img, label);SuperpixelStd(img, label)];
    Ks = RBF(im, im, sig, 2000);
    clear im;
    if min(label) == 0, label = label + 1; end
    train_label = label(trainidx);
    iP = Ks(train_label, label);
    P = P + eta_labels_size * iP;
    clear Ks;
    G = G + eta_labels_size^2 * iP(:,trainidx) * iP(:,trainidx)';
    clear iP;
end
G = (G + (P(:,trainidx) * P(:,trainidx)') .* (tau/ (lam*(eta_labels_size+1)))) ./ (lam+tau);
S = (1/(lam*(eta_labels_size+1))) .* P(:,trainidx)' / (eye(train_size) + G) * P;
pred = kcrc_predict(P(:,trainidx)'*P, P(:,trainidx)'*P(:,trainidx), S, lab, sum(P.*P));
end

function im = SuperpixelMean(img, labels)
labels = labels(:);
jdxs = sparse(bsxfun(@eq, labels, unique(labels)'));
jdxs = bsxfun(@times, jdxs, 1./sum(jdxs));
im = img * jdxs;
end

function im = SuperpixelStd(img, labels)
labels = labels(:);
jdxs = sparse(bsxfun(@eq, labels, unique(labels)'));
num = sum(jdxs);
im_mean = img * bsxfun(@times, jdxs, 1./num);
im2_mean = img.^2 * bsxfun(@times, jdxs, 1./(num-1+eps));
im = im2_mean - bsxfun(@times, im_mean.^2, num ./ (num-1+eps));
idx = num-1 ~= 0;
im = bsxfun(@times, im, double(idx));
im = max(im,0);
im = im.^0.5;
end

function labels = SuperPixelMultiScale(img, rows, cols, nCs, sigs)
lam = 0.5; dist = 0; pcs = 0;
if pcs > 0,
    img = myPCA(img,pcs)'*img;
    img = normalization(img, 0, 255)';
else
    img = normalization(img, 0, 255, 1)';
end
img = reshape(img, rows, cols, size(img,2));
labels = mex_MSERS(img,nCs,lam,sigs,dist);
end
