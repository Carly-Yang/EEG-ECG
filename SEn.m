function se=SEn(X,R,m)
% this function is referenced by original code in physionet.org
nlin=length(X);
% m=2;
nl=nlin-m;
%R=r*std(X);
cont=zeros(m+1);
for i=1:nl
    for l=(i+1):nl
        k=0;
        while k<m && abs(X(i+k)-X(l+k))<=R
            k=k+1;
            cont(k)=cont(k)+1;
        end
        if k==m && abs(X(i+m)-X(l+m))<=R
            cont(m+1)=cont(m+1)+1;
        end
    end
end
if cont(m+1)==0||cont(m)==0
    se=-log(double(1/(nl*(nl-1))));
else
    se=-log(double(cont(3)/cont(2)));
end
             
            
            
            
            
            
