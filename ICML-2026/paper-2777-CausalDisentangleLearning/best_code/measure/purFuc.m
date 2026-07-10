function [Purity]= purFuc(Y,predY)

predLidx = unique(predY); 
pred_classnum = length(predLidx);
% purity
correnum = 0;
for ci = 1:pred_classnum
    incluster = Y(predY == predLidx(ci));
%     cnub = unique(incluster);
%     inclunub = 0;
%     for cnubi = 1:length(cnub)
%         inclunub(cnubi) = length(find(incluster == cnub(cnubi)));
%     end;
    inclunub = hist(incluster, 1:max(incluster)); 
    if isempty(inclunub) 
        inclunub=0;
    end;
    correnum = correnum + max(inclunub);
end;
Purity = correnum/length(predY);