function pred = kcrc_predict(P, Q, coef, dict_lab, const)
if nargin < 5,
    const = 1;
end
cls = unique(dict_lab);
nClass = length(cls);
nTest = size(P,2);
err = zeros(nClass,nTest);
for k = 1 : nClass,
    kDict = (dict_lab == cls(k));
    err(k,:) = (sum(coef(kDict,:) .* (Q(kDict, kDict) * coef(kDict,:))) ...
        - 2*sum(coef(kDict,:).*P(kDict,:)) + const) ./ sum(coef(kDict,:) .*  coef(kDict,:));
end
[~,b] = min(err); pred = cls(b);
end