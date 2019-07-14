clear;
trainr = csvread('E:\files\SMIE\third_up\ai\BPNN_Dataset\BPNN_Dataset\train2.csv');
last_20 = csvread('E:\files\SMIE\third_up\ai\BPNN_Dataset\BPNN_Dataset\last_20.csv');
cnt = size(trainr);
len = cnt(2);
part_size = floor(cnt(1)/5*4);
train1 = trainr(1:part_size,1:len-1);
train_one = ones(part_size,1);
train = [train_one train1];
train_lable = trainr(1:part_size,len:len);
valid1 = trainr(part_size+1:cnt(1),1:len-1);
valid_one = ones(cnt(1)-part_size,1);
valid = [valid_one valid1];
valid_label = trainr(part_size+1:cnt(1),len:len);
last_label = last_20(:,len:len);
last1 = last_20(:,1:len-1);
last_1_size = size(last1);
last_one = ones(last_1_size(1),1);
last=[last_one last1];

%需要随机分割数据集%
testr = csvread('E:\files\SMIE\third_up\ai\BPNN_Dataset\BPNN_Dataset\test2.csv');
cnt_test = size(testr);
test_one = ones(cnt_test(1),1);
test =[test_one testr];

Middle = 10;
learning = 0.001;
W1 = rand(len,Middle);
W2 = rand(Middle+1,1);
loop = 1000;
E_train = zeros(loop,1);
E_valid = zeros(loop,1);
for k=1:loop;
    %forward
    hidden_input = train*W1;
    %hidden_output1 = 1.0 ./ (1.0+exp(-hidden_input));
    hidden_output1 = tanh(hidden_input);
    %加偏置
    one_hidden = ones(part_size,1);
    hidden_output = [one_hidden hidden_output1];
    %算出最终的预测结果，与原来的作比较算出误差。
    y_predict = hidden_output*W2;
    error_output = (y_predict - train_lable);
    E_train(k) = 0.5 * norm(error_output,2)* norm(error_output,2)/part_size;
    %backward，根据公式更新W1和W2
    W2_tmp = W2(2:Middle+1,:);%去掉偏置项
    W1 = W1 - learning*(train')*((error_output*W2_tmp').*(1-hidden_output1.*hidden_output1))/part_size;
    W2 = W2 - learning*(hidden_output)'*(error_output)/part_size; 
    valid_hidden_input = valid*W1;
    %valid_hidden_output1 = 1.0 ./ (1.0+exp(-valid_hidden_input));
    valid_hidden_output1=tanh(valid_hidden_input);
    %加偏置
    valid_hidden_output = [valid_one valid_hidden_output1];
    valid_y_predict = valid_hidden_output*W2;
    valid_error_out =  (valid_y_predict - valid_label);
    E_valid(k) = 0.5 * 0.5 * norm(valid_error_out,2)* norm(valid_error_out,2)/(cnt(1)-part_size);
end

 last_hidden_input =  last*W1;
    %valid_hidden_output1 = 1.0 ./ (1.0+exp(-valid_hidden_input));
    last_hidden_output1=tanh( last_hidden_input);
    %加偏置
    last_hidden_output = [ last_one  last_hidden_output1];
    last_y_predict = last_hidden_output*W2;
    
    test_hidden_input =  test*W1;
    %valid_hidden_output1 = 1.0 ./ (1.0+exp(-valid_hidden_input));
    test_hidden_output1=tanh(test_hidden_input);
    %加偏置
    test_hidden_output = [ test_one  test_hidden_output1];
    test_y_predict = test_hidden_output*W2;
    
    
figure(1);
plot(last_y_predict,'b');
hold on;
plot(last_label,'r');
figure(2);
plot(E_train,'b');
hold on;
plot(E_valid,'r');