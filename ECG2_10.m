a = load('ECG2.txt');
b = load('ECG2_1.txt');
C = vertcat(a,b);
rdet = R_detect(C(:,2),300);
rri=(rdet(2:end))./500-(rdet(1:end-1))./500;
%subplot
plot(C);hold on;plot(rdet,C(rdet),'sr') 

plot(rri); 