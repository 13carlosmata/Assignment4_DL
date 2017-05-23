function[X,Y]= ConversiontoMatrices(X_chars,Y_chars,char_to_ind,K)
X = zeros(K,size(X_chars,2));
Y = zeros(K,size(Y_chars,2));
for c=1:size(X_chars,2)
    hot_x = char_to_ind(X_chars(c));
    X(hot_x,c)=1;
    hot_y = char_to_ind(Y_chars(c));
    Y(hot_y,c)=1;
end
end
