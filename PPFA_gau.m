%for each time slot t, total 6442 smart meters,30min interval,1h 2records,1day 48records
%%
% honest_id=1002:1:1102; %100 users
respath='out/';
load('all_data.mat') %slots*user
train_len=length([0:0.5:24]); %daily load, slot 30min
train=all_data(1:train_len,:);

%each user add noise
all_users=size(all_data,2)-1; %all honest
%honest_users=round([1/3 0.5 0.6 0.7 0.8 1]*all_users);
% epsilon=[0.5 0.6 0.7 0.8 0.99];
% DP_delta=[0.001 0.005 0.01 0.05 0.1];
epsilon=0.99;
DP_delta=0.1; %(epsilon,DP_delta) DP

% honest_users=all_users;
honest_users=3000;
% honest_user=ceil(length(fileNames)/2);%at least half of all participants

%max-min each slot for all users' data
data=train(:,2:end); %1st column:slot label
% range=max(data,[],2)-min(data,[],2);
% L1_sensitivity=sum(range);
% L2_sensitivity=sqrt(sumsqr(range));
slots=size(train,1);
parties=size(train,2)-1;

%setting: 3 fog nodes, each connecting with 1000 smart meters
fog=3;
fog_sensor=1000;
%average
repeated_times=100;
MRE_delta_fog1=[];
MRE_delta_fog2=[];
MRE_delta_fog3=[];
MRE_delta_cloud=[];
%post
MRE_delta_post_fog1=[];
MRE_delta_post_fog2=[];
MRE_delta_post_fog3=[];
MRE_delta_post_cloud=[];

tic;
for p=1:length(honest_users)
    %each party sum for all slots
    slots_sum=zeros(1,honest_users(p)); 
    slots_sumsqr=zeros(1,honest_users(p)); 
    %sensitivity
    slots_max=max(data,[],2); %max each slot/row
    L1_sensitivity=sum(slots_max);
    L2_sensitivity=sqrt(sumsqr(slots_max));
    plain_sum_fog1_slots=[];
    plain_sum_fog2_slots=[];
    plain_sum_fog3_slots=[];
    plain_sum_cloud_slots=[];
    for f=1:fog
        data_slots_sensor=data(:,fog_sensor*(f-1)+1:fog_sensor*f);
        if f==1
            plain_sum_fog1_slots=sum(data_slots_sensor,2);
        end
        if f==2
            plain_sum_fog2_slots=sum(data_slots_sensor,2);
        end
        if f==3
            plain_sum_fog3_slots=sum(data_slots_sensor,2);
        end
    end
    data_cloud_slots=data(:,1:honest_users(p));
    plain_sum_cloud_slots=sum(data_cloud_slots,2);
    for delta=1:length(DP_delta)
        MRE_eps_fog1=[];
        MRE_eps_fog2=[];
        MRE_eps_fog3=[];
        MRE_eps_cloud=[];
        MRE_eps_post_fog1=[];
        MRE_eps_post_fog2=[];
        MRE_eps_post_fog3=[];
        MRE_eps_post_cloud=[];
        for eps=1:length(epsilon)
            f_epsilon=epsilon(eps)*0.8;
            s_epsilon=epsilon(eps)*0.2;
            f_delta=DP_delta(delta)*0.8;
            s_delta=DP_delta(delta)*0.2;
            data_perturbed_sum_slots_fog1_repeated=zeros(slots,repeated_times);
            data_perturbed_sum_slots_fog2_repeated=zeros(slots,repeated_times);
            data_perturbed_sum_slots_fog3_repeated=zeros(slots,repeated_times);
            data_perturbed_sum_slots_cloud_repeated=zeros(slots,repeated_times);
            for s=1:slots
                %repeat for each slot
                data_perturbed_sum_fog1_repeated=[];
                data_perturbed_sum_fog2_repeated=[];
                data_perturbed_sum_fog3_repeated=[];
                data_perturbed_sum_cloud_repeated=[];
                for r=1:repeated_times
                    for f=1:fog
                        c=sqrt(2*log(1.25/s_delta));
                        data_slot_sensor=data(s,fog_sensor*(f-1)+1:fog_sensor*f);
                        %%distributed noise generation: gaussian noise
                        sigma=c*L2_sensitivity/s_epsilon/sqrt(fog_sensor);
                        noise=sigma*randn(1,fog_sensor);
                        data_perturbed_sensor=data_slot_sensor+noise;
                        data_perturbed_sum_fog_slot(f)=sum(data_perturbed_sensor);
                        if f==1
                            data_perturbed_sum_fog1_repeated=[data_perturbed_sum_fog1_repeated data_perturbed_sum_fog_slot(f)];
                        end
                        if f==2
                            data_perturbed_sum_fog2_repeated=[data_perturbed_sum_fog2_repeated data_perturbed_sum_fog_slot(f)];
                        end
                        if f==3
                            data_perturbed_sum_fog3_repeated=[data_perturbed_sum_fog3_repeated data_perturbed_sum_fog_slot(f)];
                        end         
                    end
                    %%distributed noise generation added by fog nodes: gaussian noise
                    c=sqrt(2*log(1.25/f_delta));
                    sigma=c*L2_sensitivity/f_epsilon/sqrt(fog);
                    noise=sigma*randn(1,fog);
                    data_slot_cloud=sum(data_perturbed_sum_fog_slot+noise);
                    data_perturbed_sum_cloud_repeated=[data_perturbed_sum_cloud_repeated data_slot_cloud];
                end
                data_perturbed_sum_slots_fog1_repeated(s,:)=data_perturbed_sum_fog1_repeated;
                data_perturbed_sum_slots_fog2_repeated(s,:)=data_perturbed_sum_fog2_repeated;
                data_perturbed_sum_slots_fog3_repeated(s,:)=data_perturbed_sum_fog3_repeated;
                data_perturbed_sum_slots_cloud_repeated(s,:)=data_perturbed_sum_cloud_repeated;
            end
            %average the repeat
            mean_slots_fog1=mean(data_perturbed_sum_slots_fog1_repeated,2); 
            mean_slots_fog2=mean(data_perturbed_sum_slots_fog2_repeated,2);
            mean_slots_fog3=mean(data_perturbed_sum_slots_fog3_repeated,2);
            mean_slots_cloud=mean(data_perturbed_sum_slots_cloud_repeated,2);
            error_slots_fog1=abs(mean_slots_fog1-plain_sum_fog1_slots)./plain_sum_fog1_slots*100;
            error_slots_fog2=abs(mean_slots_fog2-plain_sum_fog2_slots)./plain_sum_fog2_slots*100;
            error_slots_fog3=abs(mean_slots_fog3-plain_sum_fog3_slots)./plain_sum_fog3_slots*100;
            error_slots_cloud=abs(mean_slots_cloud-plain_sum_cloud_slots)./plain_sum_cloud_slots*100;
            %smooth:post-processing
            data_perturbed_sum_slots_post_fog1=smooth(mean_slots_fog1,5);
            error_slots_post_fog1=abs(data_perturbed_sum_slots_post_fog1-plain_sum_fog1_slots)./plain_sum_fog1_slots*100;
            data_perturbed_sum_slots_post_fog2=smooth(mean_slots_fog2,5);
            error_slots_post_fog2=abs(data_perturbed_sum_slots_post_fog2-plain_sum_fog2_slots)./plain_sum_fog2_slots*100;
            data_perturbed_sum_slots_post_fog3=smooth(mean_slots_fog3,5);
            error_slots_post_fog3=abs(data_perturbed_sum_slots_post_fog3-plain_sum_fog3_slots)./plain_sum_fog3_slots*100;
            data_perturbed_sum_slots_post_cloud=smooth(mean_slots_cloud,5);
            error_slots_post_cloud=abs(data_perturbed_sum_slots_post_cloud-plain_sum_cloud_slots)./plain_sum_cloud_slots*100;
            %mean error for all slots for epsilon=[0.5 0.6 0.7 0.8 0.99],honest_users=all_users;
            MRE_eps_fog1=[MRE_eps_fog1;mean(error_slots_fog1)];
            MRE_eps_fog2=[MRE_eps_fog2;mean(error_slots_fog2)];
            MRE_eps_fog3=[MRE_eps_fog3;mean(error_slots_fog3)];
            MRE_eps_cloud=[MRE_eps_cloud;mean(error_slots_cloud)];
            %post
            MRE_eps_post_fog1=[MRE_eps_post_fog1;mean(error_slots_post_fog1)];
            MRE_eps_post_fog2=[MRE_eps_post_fog2;mean(error_slots_post_fog2)];
            MRE_eps_post_fog3=[MRE_eps_post_fog3;mean(error_slots_post_fog3)];
            MRE_eps_post_cloud=[MRE_eps_post_cloud;mean(error_slots_post_cloud)];
        end
        MRE_delta_fog1=[MRE_delta_fog1 MRE_eps_fog1];
        MRE_delta_fog2=[MRE_delta_fog2 MRE_eps_fog2];
        MRE_delta_fog3=[MRE_delta_fog3 MRE_eps_fog3];
        MRE_delta_cloud=[MRE_delta_cloud MRE_eps_cloud];
        %post
        MRE_delta_post_fog1=[MRE_delta_post_fog1 MRE_eps_post_fog1];
        MRE_delta_post_fog2=[MRE_delta_post_fog2 MRE_eps_post_fog2];
        MRE_delta_post_fog3=[MRE_delta_post_fog3 MRE_eps_post_fog3];
        MRE_delta_post_cloud=[MRE_delta_post_cloud MRE_eps_post_cloud];
    end
end
time=toc;
save(strcat(respath,'time_gau_epsilon_repeat100'),'time');
%mean error vs time
error_gua_fog1=mean(error_slots_fog1);
error_gua_fog2=mean(error_slots_fog2);
error_gua_fog3=mean(error_slots_fog3);
error_gua_cloud=mean(error_slots_cloud);
save(strcat(respath,'error_gua_fog1'),'error_gua_fog1');
save(strcat(respath,'error_gua_fog2'),'error_gua_fog2');
save(strcat(respath,'error_gua_fog3'),'error_gua_fog3');
save(strcat(respath,'error_gua_cloud'),'error_gua_cloud');
%post mean error vs time
error_gua_post_fog1=mean(error_slots_post_fog1);
error_gua_post_fog2=mean(error_slots_post_fog2);
error_gua_post_fog3=mean(error_slots_post_fog3);
error_gua_post_cloud=mean(error_slots_post_cloud);
save(strcat(respath,'error_gua_post_fog1'),'error_gua_post_fog1');
save(strcat(respath,'error_gua_post_fog2'),'error_gua_post_fog2');
save(strcat(respath,'error_gua_post_fog3'),'error_gua_post_fog3');
save(strcat(respath,'error_gua_post_cloud'),'error_gua_post_cloud');

%%
%%Aggregated value-slots, draw raw data/ PP/ post-processing reconstructed data, one day: 49 records
%% epsilon=0.5, honest_users=all_users;
% figure
% time_slots=[0:0.5:24];
% plot(time_slots, plain_sum, '-*', 'LineWidth', 2, 'MarkerSize', 6); 
% hold on;
% plot(time_slots, mean_slots, '-+', 'LineWidth', 2, 'MarkerSize', 6);
% hold on;
% plot(time_slots, data_perturbed_sum_slots_post, '-*', 'LineWidth', 2, 'MarkerSize', 6);
% xlim([0 24])
% xlabel('Daily time slot[h] (\epsilon=0.5)','fontweight','bold','fontsize',24);
% ylabel('Aggregated value','fontweight','bold','fontsize',24);
% methods={'Original aggregation','Perturbed aggregation','Smoothed perturbed aggregation'};
% legend(methods, 'location', 'southeast','fontsize',20);
% no smooth
%fog1
figure
time_slots=[0:0.5:24];
plot(time_slots, plain_sum_fog1_slots, '-*', 'LineWidth', 2, 'MarkerSize', 6); 
hold on;
plot(time_slots, mean_slots_fog1, '-+', 'LineWidth', 2, 'MarkerSize', 6);
xlim([0 24])
ylim([0 1000])
xlabel('Daily time slot[h] (\epsilon=0.99, \delta=0.1)','fontweight','bold','fontsize',24);
set(gca,'FontSize',24)
%set(gca,'XTickLabel',time_slots,'FontSize',24)
ylabel('Aggregation at Fog node 1','fontweight','bold','fontsize',24);
methods={'Original aggregation','Perturbed aggregation'};
legend(methods, 'location', 'southeast','fontsize',20);
title('Gaussian noise','fontweight','bold','fontsize',24);
%fog2
figure
time_slots=[0:0.5:24];
plot(time_slots, plain_sum_fog2_slots, '-*', 'LineWidth', 2, 'MarkerSize', 6); 
hold on;
plot(time_slots, mean_slots_fog2, '-+', 'LineWidth', 2, 'MarkerSize', 6);
xlim([0 24])
ylim([0 1000])
xlabel('Daily time slot[h] (\epsilon=0.99, \delta=0.1)','fontweight','bold','fontsize',24);
set(gca,'FontSize',24)
%set(gca,'XTickLabel',time_slots,'FontSize',24)
ylabel('Aggregation at Fog node 2','fontweight','bold','fontsize',24);
methods={'Original aggregation','Perturbed aggregation'};
legend(methods, 'location', 'southeast','fontsize',20);
title('Gaussian noise','fontweight','bold','fontsize',24);

%fog3
figure
time_slots=[0:0.5:24];
plot(time_slots, plain_sum_fog3_slots, '-*', 'LineWidth', 2, 'MarkerSize', 6); 
hold on;
plot(time_slots, mean_slots_fog3, '-+', 'LineWidth', 2, 'MarkerSize', 6);
xlim([0 24])
ylim([0 1000])
xlabel('Daily time slot[h] (\epsilon=0.99, \delta=0.1)','fontweight','bold','fontsize',24);
set(gca,'FontSize',24)
%set(gca,'XTickLabel',time_slots,'FontSize',24)
ylabel('Aggregation at Fog node 3','fontweight','bold','fontsize',24);
methods={'Original aggregation','Perturbed aggregation'};
legend(methods, 'location', 'southeast','fontsize',20);
title('Gaussian noise','fontweight','bold','fontsize',24);

%cloud
figure
time_slots=[0:0.5:24];
plot(time_slots, plain_sum_cloud_slots, '-*', 'LineWidth', 2, 'MarkerSize', 6); 
hold on;
plot(time_slots, mean_slots_cloud, '-+', 'LineWidth', 2, 'MarkerSize', 6);
xlim([0 24])
ylim([0 2500])
xlabel('Daily time slot[h] (\epsilon=0.99, \delta=0.1)','fontweight','bold','fontsize',24);
set(gca,'FontSize',24)
%set(gca,'XTickLabel',time_slots,'FontSize',24)
ylabel('Aggregation at Cloud','fontweight','bold','fontsize',24);
methods={'Original aggregation','Perturbed aggregation'};
legend(methods, 'location', 'southeast','fontsize',20);
title('Gaussian noise','fontweight','bold','fontsize',24);

%%
%error-slots, epsilon=0.5, honest_users=all_users;
%fog1
figure
plot(time_slots, error_slots_fog1, '-*', 'LineWidth', 2, 'MarkerSize', 6); 
hold on;
plot(time_slots, error_slots_post_fog1, '-+', 'LineWidth', 2, 'MarkerSize', 6);
xlim([0 24])
ylim([0 80])
set(gca,'FontSize',24)
xlabel('Daily time slot[h] (\epsilon=0.99, \delta=0.1)','fontweight','bold','fontsize',24);
ylabel('Fog node 1 MRE(%)','fontweight','bold','fontsize',24);
methods={'DP aggregation MRE','Smoothed DP aggregation MRE'};
legend(methods, 'location', 'northeast','fontsize',20);
title('Gaussian noise','fontweight','bold','fontsize',24);

%no smooth
% figure
% time_slots=[0:0.5:24];
% plot(time_slots, error_slots_fog1, '-+', 'LineWidth', 2, 'MarkerSize', 6); 
% set(gca,'FontSize',24)
% %set(gca,'XTickLabel',time_slots,'FontSize',24)
% xlabel('Daily time slot[h] (\epsilon=0.5, \delta=0.1)','fontweight','bold','fontsize',24);
% ylabel('Mean relative error %','fontweight','bold','fontsize',24);
%fog2
figure
plot(time_slots, error_slots_fog2, '-*', 'LineWidth', 2, 'MarkerSize', 6); 
hold on;
plot(time_slots, error_slots_post_fog2, '-+', 'LineWidth', 2, 'MarkerSize', 6);
xlim([0 24])
ylim([0 80])
set(gca,'FontSize',24)
xlabel('Daily time slot[h] (\epsilon=0.99, \delta=0.1)','fontweight','bold','fontsize',24);
ylabel('Fog node 2 MRE(%)','fontweight','bold','fontsize',24);
methods={'DP aggregation MRE','Smoothed DP aggregation MRE'};
legend(methods, 'location', 'northeast','fontsize',20);
title('Gaussian noise','fontweight','bold','fontsize',24);
%no smooth
% figure
% time_slots=[0:0.5:24];
% plot(time_slots, error_slots_fog2, '-+', 'LineWidth', 2, 'MarkerSize', 6); 
% set(gca,'FontSize',24)
% %set(gca,'XTickLabel',time_slots,'FontSize',24)
% xlabel('Daily time slot[h] (\epsilon=0.5, \delta=0.1)','fontweight','bold','fontsize',24);
% ylabel('Mean relative error %','fontweight','bold','fontsize',24)
%fog3
figure
plot(time_slots, error_slots_fog3, '-*', 'LineWidth', 2, 'MarkerSize', 6); 
hold on;
plot(time_slots, error_slots_post_fog3, '-+', 'LineWidth', 2, 'MarkerSize', 6);
xlim([0 24])
ylim([0 80])
set(gca,'FontSize',24)
xlabel('Daily time slot[h] (\epsilon=0.99, \delta=0.1)','fontweight','bold','fontsize',24);
ylabel('Fog node 3 MRE(%)','fontweight','bold','fontsize',24);
methods={'DP aggregation MRE','Smoothed DP aggregation MRE'};
legend(methods, 'location', 'northeast','fontsize',20);
title('Gaussian noise','fontweight','bold','fontsize',24);
%no smooth
% figure
% time_slots=[0:0.5:24];
% plot(time_slots, error_slots_fog3, '-+', 'LineWidth', 2, 'MarkerSize', 6); 
% set(gca,'FontSize',24)
% %set(gca,'XTickLabel',time_slots,'FontSize',24)
% xlabel('Daily time slot[h] (\epsilon=0.5, \delta=0.1)','fontweight','bold','fontsize',24);
% ylabel('Mean relative error %','fontweight','bold','fontsize',24)
%cloud
figure
plot(time_slots, error_slots_cloud, '-*', 'LineWidth', 2, 'MarkerSize', 6); 
hold on;
plot(time_slots, error_slots_post_cloud, '-+', 'LineWidth', 2, 'MarkerSize', 6);
xlim([0 24])
ylim([0 80])
set(gca,'FontSize',24)
xlabel('Daily time slot[h] (\epsilon=0.99, \delta=0.1)','fontweight','bold','fontsize',24);
ylabel('Cloud MRE(%)','fontweight','bold','fontsize',24);
methods={'DP aggregation MRE','Smoothed DP aggregation MRE'};
legend(methods, 'location', 'northeast','fontsize',20);
title('Gaussian noise','fontweight','bold','fontsize',24);
%no smooth
% figure
% time_slots=[0:0.5:24];
% plot(time_slots, error_slots_cloud, '-+', 'LineWidth', 2, 'MarkerSize', 6); 
% set(gca,'FontSize',24)
% %set(gca,'XTickLabel',time_slots,'FontSize',24)
% xlabel('Daily time slot[h] (\epsilon=0.5, \delta=0.1)','fontweight','bold','fontsize',24);
% ylabel('Mean relative error %','fontweight','bold','fontsize',24)
%%
%MRE-eps
%%epsilon=[0.5 0.6 0.7 0.8 0.99];honest_users=all_users;
%fog1
figure
% plot(epsilon, MRE_eps, '-+', 'LineWidth', 2, 'MarkerSize', 8); 
% hold on;
% plot(epsilon, MRE_eps_post, '-*', 'LineWidth', 2, 'MarkerSize', 8); 
% xlabel('\epsilon','fontweight','bold','fontsize',24);
% ylabel('Mean relative error %','fontweight','bold','fontsize',24);
% epsilon_error={'Perturbed aggregation error','Smoothed perturbed aggregation error'};
% legend(epsilon_error, 'location', 'northeast','fontsize',20);
plot(epsilon, MRE_eps_fog1, '-*', 'LineWidth', 2, 'MarkerSize', 6);  
hold on;
plot(epsilon, MRE_eps_post_fog1, '-+', 'LineWidth', 2, 'MarkerSize', 6);
ylim([0 25])
set(gca,'FontSize',24)
xlabel('Privacy budget \epsilon (\delta=0.1)','fontweight','bold','fontsize',24);
ylabel('Fog node 1 MRE(%)','fontweight','bold','fontsize',24);
methods={'DP aggregation MRE','Smoothed DP aggregation MRE'};
legend(methods, 'location', 'northeast','fontsize',20);
title('Gaussian noise','fontweight','bold','fontsize',24);
%fog2
figure
plot(epsilon, MRE_eps_fog2, '-*', 'LineWidth', 2, 'MarkerSize', 6);
hold on;
plot(epsilon, MRE_eps_post_fog2, '-+', 'LineWidth', 2, 'MarkerSize', 6);
ylim([0 25])
set(gca,'FontSize',24)
xlabel('Privacy budget \epsilon (\delta=0.1)','fontweight','bold','fontsize',24);
ylabel('Fog node 2 MRE(%)','fontweight','bold','fontsize',24);
methods={'DP aggregation MRE','Smoothed DP aggregation MRE'};
legend(methods, 'location', 'northeast','fontsize',20);
title('Gaussian noise','fontweight','bold','fontsize',24);
%fog3
figure
plot(epsilon, MRE_eps_fog3, '-*', 'LineWidth', 2, 'MarkerSize', 6);  
hold on;
plot(epsilon, MRE_eps_post_fog3, '-+', 'LineWidth', 2, 'MarkerSize', 6);
ylim([0 25])
set(gca,'FontSize',24)
xlabel('Privacy budget \epsilon (\delta=0.1)','fontweight','bold','fontsize',24);
ylabel('Fog node 3 MRE(%)','fontweight','bold','fontsize',24);
methods={'DP aggregation MRE','Smoothed DP aggregation MRE'};
legend(methods, 'location', 'northeast','fontsize',20);
title('Gaussian noise','fontweight','bold','fontsize',24);
%cloud
figure
plot(epsilon, MRE_eps_cloud, '-*', 'LineWidth', 2, 'MarkerSize', 6);  
hold on;
plot(epsilon, MRE_eps_post_cloud, '-+', 'LineWidth', 2, 'MarkerSize', 6);
ylim([0 25])
set(gca,'FontSize',24)
xlabel('Privacy budget \epsilon (\delta=0.1)','fontweight','bold','fontsize',24);
ylabel('Cloud MRE(%)','fontweight','bold','fontsize',24);
methods={'DP aggregation MRE','Smoothed DP aggregation MRE'};
legend(methods, 'location', 'northeast','fontsize',20);
title('Gaussian noise','fontweight','bold','fontsize',24);
%%
%MRE-delta
%%epsilon=[0.5 0.6 0.7 0.8 0.99];honest_users=all_users;
%fog1
figure
plot(DP_delta, MRE_delta_fog1, '-*', 'LineWidth', 2, 'MarkerSize', 6);  
hold on;
plot(DP_delta, MRE_delta_post_fog1, '-+', 'LineWidth', 2, 'MarkerSize', 6);
ylim([0 50])
set(gca,'Xscale','log','XTick',[DP_delta],'XTickLabel',DP_delta,'FontSize',24)
xlabel('Relaxed probability \delta (\epsilon=0.5)','fontweight','bold','fontsize',24);
ylabel('Fog node 1 MRE(%)','fontweight','bold','fontsize',24);
methods={'DP aggregation MRE','Smoothed DP aggregation MRE'};
legend(methods, 'location', 'northeast','fontsize',20);
title('Gaussian noise','fontweight','bold','fontsize',24);
%fog2
figure
plot(DP_delta, MRE_delta_fog2, '-*', 'LineWidth', 2, 'MarkerSize', 6);
hold on;
plot(DP_delta, MRE_delta_post_fog2, '-+', 'LineWidth', 2, 'MarkerSize', 6);
ylim([0 50])
set(gca,'Xscale','log','XTick',[DP_delta],'XTickLabel',DP_delta,'FontSize',24)
xlabel('Relaxed probability \delta (\epsilon=0.5)','fontweight','bold','fontsize',24);
ylabel('Fog node 2 MRE(%)','fontweight','bold','fontsize',24);
methods={'DP aggregation MRE','Smoothed DP aggregation MRE'};
legend(methods, 'location', 'northeast','fontsize',20);
title('Gaussian noise','fontweight','bold','fontsize',24);
%fog3
figure
plot(DP_delta, MRE_delta_fog3, '-*', 'LineWidth', 2, 'MarkerSize', 6);  
hold on;
plot(DP_delta, MRE_delta_post_fog3, '-+', 'LineWidth', 2, 'MarkerSize', 6);
ylim([0 50])
set(gca,'Xscale','log','XTick',[DP_delta],'XTickLabel',DP_delta,'FontSize',24)
xlabel('Relaxed probability \delta (\epsilon=0.5)','fontweight','bold','fontsize',24);
ylabel('Fog node 3 MRE(%)','fontweight','bold','fontsize',24);
methods={'DP aggregation MRE','Smoothed DP aggregation MRE'};
legend(methods, 'location', 'northeast','fontsize',20);
title('Gaussian noise','fontweight','bold','fontsize',24);
%cloud
figure
plot(DP_delta, MRE_delta_cloud, '-*', 'LineWidth', 2, 'MarkerSize', 6);  
hold on;
plot(DP_delta, MRE_delta_post_cloud, '-+', 'LineWidth', 2, 'MarkerSize', 6);
ylim([0 50])
set(gca,'Xscale','log','XTick',[DP_delta],'XTickLabel',DP_delta,'FontSize',24)
xlabel('Relaxed probability \delta (\epsilon=0.5)','fontweight','bold','fontsize',24);
ylabel('Cloud MRE(%)','fontweight','bold','fontsize',24);
methods={'DP aggregation MRE','Smoothed DP aggregation MRE'};
legend(methods, 'location', 'northeast','fontsize',20);
title('Gaussian noise','fontweight','bold','fontsize',24);
%%
%MRE-users
%honest_users=round([1/3 0.5 0.6 0.7 0.8 1]*all_users);epsilon=0.5;
% figure
% % plot(honest_users, MRE_users, '-+', 'LineWidth', 2, 'MarkerSize', 8); 
% % hold on;
% % plot(honest_users, MRE_users_post, '-*', 'LineWidth', 2, 'MarkerSize', 8); 
% % xlabel('\epsilon','fontweight','bold','fontsize',24);
% % ylabel('Mean relative error %','fontweight','bold','fontsize',24);
% % users_error={'Perturbed aggregation error','Smoothed perturbed aggregation error'};
% % legend(users_error, 'location', 'northeast','fontsize',20);
% plot(honest_users, MRE_users, '-+', 'LineWidth', 2, 'MarkerSize', 8); 
% %set(gca,'FontSize',24)
% set(gca,'Xscale','log','XTick',[honest_users],'XTickLabel',honest_users,'fontsize',24)
% xlabel('Number of honest parties n (\epsilon=0.5, \delta=0.1)','fontweight','bold','fontsize',24);
% ylabel('Mean relative error %','fontweight','bold','fontsize',24);