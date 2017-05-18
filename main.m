close all
clear all
clc
%%
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
char_to_ind = containers.Map(array_chars,int32([1:K]));
ind_to_char = containers.Map(int32([1:K]),array_chars);

%%
m=100;
eta = .1;   %learning rate
seq_length = 25;    %length of the input sequences
sig = .01;
RNN = RNNmodel(m,K,sig);
%%
n = 25;
h0 = randn(m,1); %hidden state at time 0
x_t0 = rand(83,n); % first dummy input vector to the RNN  --- ???
at = {}; ht = {}; ot = {}; pt = {};
X_ixs = [];
for i=1:n
    X_ixs = [X_ixs, x_t0];
    [at{i},ht{i},ot{i},pt{i},xt]= synthesize(RNN,h0,x_t0,n);
    x_t0(2) = xt;
end
fprintf("done \n")


