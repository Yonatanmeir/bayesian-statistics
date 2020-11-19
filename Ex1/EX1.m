%% main :
%% figure 5.2
clear all; clc;
fig_counter=1
for monte_carlo_i=1:2
    if monte_carlo_i==1
        data=[0 0 0 1];
    else
        data=[ones(10,1); zeros(30,1)];
    end
theta=linspace(0,1,1001);
ptheta=min([theta; 1-theta]);
ptheta=ptheta/sum(ptheta);
figure(fig_counter);
subplot(3,1,1); %prior 
plot(theta,ptheta);
xlabel('\theta');
ylabel('p(\theta)');
title('prior');
[~,mode]=max(ptheta);
hdiLim = mbe_hdi(ptheta,0.95);
theta_lb=theta(find(ptheta>hdiLim(1),1));
theta_rb=find(ptheta>hdiLim(1));
theta_rb=theta(theta_rb(end));
line([theta_lb theta_rb],[hdiLim(1),hdiLim(1)],'color','k')
legend(['mode=' num2str(theta(mode))],'95 % HDI');

subplot(3,1,2);%likelihood 
p_data_given_theta=theta.^(sum(data)).*(1-theta).^(length(data)-sum(data));
plot(theta,p_data_given_theta);
xlabel('\theta');
ylabel('p(D|\theta)');
title('likelihood');
[~,mode]=max(p_data_given_theta);
legend(['mode=' num2str(theta(mode)) ' Z=' num2str(sum(data)) ' N=' num2str(length(data))]);

subplot(3,1,3);%posterior 1
posterior=ptheta.*p_data_given_theta/sum(p_data_given_theta.*ptheta);
plot(theta,posterior);
xlabel('\theta');
ylabel('p(\theta|D)');
title('posterior');
[~,mode]=max(posterior);
hdiLim = mbe_hdi(posterior,0.95);
theta_lb=theta(find(posterior>hdiLim(1),1));
theta_rb=find(posterior>hdiLim(1));
theta_rb=theta(theta_rb(end));
line([theta_lb theta_rb],[hdiLim(1),hdiLim(1)],'color','k')
legend(['mode=' num2str(theta(mode))],'95 % HDI');
legend(['mode=' num2str(theta(mode))]);
fig_counter=fig_counter+1
end

%% figure 5.3
%clear all; clc;
for monte_carlo_i=1:2
    if monte_carlo_i==1
        %generalized normal distrbution:
        beta=8;
        alpha=0.2;
        mu=0.5;
        theta=linspace(0,1,1001);
        ptheta=beta/(2*alpha*gamma(1/beta))*exp(-((abs(theta-mu)/alpha).*beta));
        ptheta=ptheta/sum(ptheta);
        data=[0 0 0 1];
    else
        beta=0.5;
        alpha=0.2;
        mu=0.5;
        theta=linspace(0,1,1001);
        ptheta=beta/(2*alpha*gamma(1/beta))*exp(-(abs(theta-mu)/alpha).*beta);
        ptheta=ptheta/sum(ptheta);
        data=[ones(10,1); zeros(30,1)];
    end
%theta=linspace(0,1,1001);
%ptheta=min([theta; 1-theta]);
%ptheta=ptheta/sum(ptheta);
figure(fig_counter);
subplot(3,1,1); %prior 
plot(theta,ptheta);
xlabel('\theta');
ylabel('p(\theta)');
title('prior');
[~,mode]=max(ptheta);
hdiLim = mbe_hdi(ptheta,0.95);
theta_lb=theta(find(ptheta>hdiLim(1),1));
theta_rb=find(ptheta>hdiLim(1));
theta_rb=theta(theta_rb(end));
line([theta_lb theta_rb],[hdiLim(1),hdiLim(1)],'color','k')
legend(['mode=' num2str(theta(mode))],'95 % HDI');

subplot(3,1,2);%likelihood 
p_data_given_theta=theta.^(sum(data)).*(1-theta).^(length(data)-sum(data));
plot(theta,p_data_given_theta);
xlabel('\theta');
ylabel('p(D|\theta)');
title('likelihood');
[~,mode]=max(p_data_given_theta);
legend(['mode=' num2str(theta(mode)) ' Z=' num2str(sum(data)) ' N=' num2str(length(data))]);

subplot(3,1,3);%posterior 1
posterior=ptheta.*p_data_given_theta/sum(p_data_given_theta.*ptheta);
plot(theta,posterior);
xlabel('\theta');
ylabel('p(\theta|D)');
title('posterior');
[~,mode]=max(posterior);
hdiLim = mbe_hdi(posterior,0.95);
theta_lb=theta(find(posterior>hdiLim(1),1));
theta_rb=find(posterior>hdiLim(1));
theta_rb=theta(theta_rb(end));
line([theta_lb theta_rb],[hdiLim(1),hdiLim(1)],'color','k')
legend(['mode=' num2str(theta(mode))],'95 % HDI');
fig_counter=fig_counter+1
end

%% figure 6.3
data=[1 zeros(1,9)]
theta=linspace(0,1,1001);
ptheta=betapdf(theta,5,5)
%ptheta=ptheta/sum(ptheta);
figure(fig_counter);
subplot(3,1,1); %prior 
plot(theta,ptheta);
xlabel('\theta');
ylabel('p(\theta)|a=5,b=5)');
title('prior');
[~,mode]=max(ptheta);
hdiLim = mbe_hdi(ptheta,0.95);
theta_lb=theta(find(ptheta>hdiLim(1),1));
theta_rb=find(ptheta>hdiLim(1));
theta_rb=theta(theta_rb(end));
line([theta_lb theta_rb],[hdiLim(1),hdiLim(1)],'color','k')
legend(['mean=' num2str(0.5)],'95 % HDI');

subplot(3,1,2);%likelihood 
p_data_given_theta=theta.^(sum(data)).*(1-theta).^(length(data)-sum(data));
plot(theta,p_data_given_theta);
xlabel('\theta');
ylabel('p(D|\theta)');
title('likelihood');
[~,mode]=max(p_data_given_theta);
legend(['Z=' num2str(sum(data)) ' N=' num2str(length(data))]);

subplot(3,1,3);%posterior 1
posterior=ptheta.*p_data_given_theta/sum(p_data_given_theta.*ptheta);
plot(theta,posterior);
xlabel('\theta');
ylabel('p(\theta|D)');
title('posterior');
[~,mode]=max(posterior);
hdiLim = mbe_hdi(posterior,0.95);
theta_lb=theta(find(posterior>hdiLim(1),1));
theta_rb=find(posterior>hdiLim(1));
theta_rb=theta(theta_rb(end));
line([theta_lb theta_rb],[hdiLim(1),hdiLim(1)],'color','k')
legend(['mode=' num2str(theta(mode))],'95 % HDI');
fig_counter=fig_counter+1