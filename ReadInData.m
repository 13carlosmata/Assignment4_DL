function [char_to_ind, ind_to_char,K]  = ReadInData(file)

%Reading the data
% book_fname = 'goblet_book.txt';
book_fname = file;
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

end