close all
clear all
clc

filename = 'goblet_book.txt';
% [char_to_ind, ind_to_char,K] = ReadInData(filename);
book_fname = filename;
fid = fopen(book_fname,'r');
book_data = fscanf(fid,'%c');
fclose(fid);
book_chars = unique(book_data)';
K = length(book_chars); 
array_chars = {};
for i=1:K
    array_chars{i}=book_chars(i);
end
char_to_ind = containers.Map('KeyType','char','ValueType','int32');
ind_to_char = containers.Map('KeyType','int32','ValueType','char');
char_to_ind = containers.Map(array_chars,int32(1:K));
ind_to_char = containers.Map(int32(1:K),array_chars);

%%
m=100;
eta = .1;   %learning rate
seq_length = 25;    %length of the input sequences
sig = .01;
RNN = RNNmodel(m,K,sig);
X_chars= book_data(1:seq_length);
Y_chars = book_data(2:seq_length+1);
h = zeros(m,1); %hidden state at time 0




%%   All this is the synthesizer
% x_t0 = rand(83,n); % first dummy input vector to the RNN  --- ???
% at = {}; ht = {}; ot = {}; pt = {};
% I = [];
% Yix = [];
% for t=1:seq_length
%     [at{t},ht{t},ot{t},pt{t},ii]= synthesize(RNN,h0,x_t0);
%     I = [I;ii];
%     Yix(ii,t) = 1; 
% end
% last_text='';
% for ind=1:seq_length
%     chars=ind_to_char(I(ind));
%     last_text=[last_text,chars];
% end


%% Conversion to matrices
[X,Y]= ConversiontoMatrices(X_chars,Y_chars,char_to_ind,K);
[A,H,O,P,I]= synthesize(RNN,h,X);
[grads,h] = Gradients(RNN,h,X,Y);
num_grads = ComputeGradsNum(X, Y, RNN, 1e-4);

fprintf("\n done \n")





