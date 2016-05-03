%%
mex -setup;

%% demonstrates mono visual odometry on an image sequence
disp('===========================');
clear; close all; dbstop error; clc;


img_dir     = 'D:\VO\data_odometry_gray\dataset\sequences\03\image_0';

param.f      = 721.5;
param.cu     = 609.5;
param.cv     = 172.8;
param.height = 1.6;
param.pitch  = -0.08;
first_frame  = 0;
last_frame   = 800;

% init visual odometry
visualOdometryMonoMex('init',param);

% init transformation matrix array
Tr_total{1} = eye(4);

% create figure
figure('Color',[1 1 1]);
ha1 = axes('Position',[0.05,0.7,0.9,0.25]);
axis off;
ha2 = axes('Position',[0.05,0.05,0.9,0.6]);
set(gca,'XTick',-500:10:500);
set(gca,'YTick',-500:10:500);
axis equal, grid on, hold on;


% for all frames do
replace = 0;

% >>>>>>>>START: EKF PARAMETERS<<<<<<<<<<<<<
    NN = last_frame - first_frame + 1;
    
    load('D:\VO\data_odometry_gray\dataset\sequences\03\03.txt');
    gtdata = X03';
    for i=1:NN
       gtEulAng(:,i) = rotm2eul([X03(i,1:3); X03(i,5:7); X03(i,9:11)]);
    end
    gt = [gtdata(4,1:NN);gtdata(12,1:NN); gtEulAng(3,1:NN)];
    
    n_states=35;
    q_proc=0.1;                                 %std of process 
    r_meas=0.1;                                 %std of measurement
    n_meas_states = 3;                          % x, y, theta
    Q_proc=q_proc^2*eye(n_states);              % covariance of process
    R_meas=r_meas^2*eye(n_meas_states);         % covariance of measurement  
    dt = 0.1;
    wb=2.71 ;Jz=1; m=1454; a=1; b=1; h = 1; tf = 1; tr = 1; ckFL=1; ckFR = 1; ckRL=1; ckRR=1; caFL=1; caFR=1; caRL=1; caRR=1; wheelr =1;
    delta = 1; wFL = 1; wFR =1; wRL=1; wRR=1;
    
    % y_handle=@(x)[x(2)/dt;x(4)/dt;x(6)/dt];      % measurement equation
    y_handle=@(x)[x(33);x(34);x(35)];

    s_state=[gt(1,1) ;0;gt(1,2);0;gt(1,3);0];         % initial state
    x=s_state+q_proc*randn(n_states,1);               % initial state with noise
    P = 0.1*eye(n_states);                            % initial state covraiance
    x_store = zeros(n_states,NN);                     %estmate    
    meas_store = zeros(n_meas_states,NN);             %actual
  
  
% >>>>>>>>>>>>>END: EKF PARAMTERS<<<<<<<<<<<<<<


for frame=first_frame:last_frame
  
  % 1-based index
  k = frame-first_frame+1;
  
  % read current images
  %   I = imread([img_dir '/I1_' num2str(frame,'%06d') '.png']);
  I = imread([img_dir '/' num2str(frame,'%06d') '.png']);

  % compute egomotion
  Tr = visualOdometryMonoMex('process',I,replace);
 
  % accumulate egomotion, starting with second frame
  if k>1
    
    % if motion estimate failed: set replace "current frame" to "yes"
    % this will cause the "last frame" in the ring buffer unchanged
    if isempty(Tr)
      replace = 1;
      Tr_total{k} = Tr_total{k-1};
      eulAng(:,k) = rotm2eul(Tr_total{k}(1:3,1:3));
      
    % on success: update total motion (=pose)
    else
      replace = 0;
      Tr_total{k} = Tr_total{k-1}*inv(Tr);
      eulAng(:,k) = rotm2eul(Tr_total{k}(1:3,1:3));
    end
  end

  % update image
  axes(ha1); cla;
  imagesc(I); colormap(gray);
  axis off;
  
  % update trajectory
  axes(ha2);
  
  
  if k>1
    plot([Tr_total{k-1}(1,4) Tr_total{k}(1,4)], ...
         [Tr_total{k-1}(3,4) Tr_total{k}(3,4)],'-xb','LineWidth',1);
      
      
      meas = [Tr_total{k}(1,4); Tr_total{k}(3,4);eulAng(3,k)];
      meas_store(:,k)= meas;                            % save actual state
      
      % f = [Vx Vy ax ay yawA beta betaR yawR alphaFL alphaFR aplphaRL alphaRR fzFL fzFR fzRL fzRR sFL sFR sRL sRR vFL vFR vRL vRR fxFL fxFR fxRL fxRR fyFL fyFR fyRL fyRR x y yaw]
      f=@(x)[x(1)+x(3)*dt;...     % 1. Vx  
        x(2)+x(4)*dt;...          % 2. Vy
        (1/m)*((x(25)*cos(delta))-(x(29)*sin(delta))+x(26)*cos(delta)-x(30)*sin(delta)+x(27)+x(28));...       % 3. ax
        (1/m)*((x(29)*cos(delta))+(x(25)*sin(delta))+x(30)*cos(delta)+x(26)*sin(delta)+x(31)+x(32));...       % 4. ay
        (1/Jz)*(((tf/2)*(x(25)*cos(delta)-x(29)*sin(delta)-x(26)*cos(delta)+x(30)*sin(delta)))+((tr/2)*(x(27)-x(28)))+(a*(x(29)*cos(delta)+x(25)*sin(delta)+x(30)*cos(delta)+x(26)*sin(delta)))-(b*(x(31)+x(32)))+ mzFL + mzFR + mzRL + mzRR);... % 5. yawA
        atan(x(2)/x(1));...     % 6. beta
        (1/m*sqrt(((x(1))^2 + (x(2))^2)))*(cos(x(6))*(cos(delta)*(x(25)+x(26)) - sin(delta)*(x(29)+x(30))) - sin(x(6))*(sin(delta)*(x(25)+x(26)) - sin(delta)*(x(29)+x(30))));...   % 7. betaR
        sqrt(((x(1))^2 + (x(2))^2))*tan(delta)/wb;...        % 8. yawR
        delta-atan((x(2)+a*x(8))/(x(1)+tf*(x(8)/2)));...     % 9. alphaFL
        delta-atan((x(2)+a*x(8))/(x(1)-tf*(x(8)/2)));...     % 10. alphaFR
        atan((-x(2)+b*x(8))/(x(1)+tr*(x(8)/2)));...          % 11. alphaRL
        atan((-x(2)+b*x(8))/(x(1)-tr*(x(8)/2)));...          % 12. alphaRR
        (0.5*m*g + (m*x(4)*h)/tf)*(b/(a+b)) - 0.5*m*x(3)*(h/a+b);...        % 13. fzFL
        (0.5*m*g - (m*x(4)*h)/tf)*(b/(a+b)) - 0.5*m*x(3)*(h/a+b);...        % 14. fzFR
        (0.5*m*g + (m*x(4)*h)/tr)*(b/(a+b)) + 0.5*m*x(3)*(h/a+b);...        % 15. fzRL
        (0.5*m*g - (m*x(4)*h)/tr)*(b/(a+b)) + 0.5*m*x(3)*(h/a+b);...        % 16. fzRR
        ((wFL*wheelr)/x(21))-1;...      % 17. sFL
        ((wFR*wheelr)/x(22))-1;...      % 18. sFR
        ((wRL*wheelr)/x(23))-1;...      % 19. sRL
        ((wRR*wheelr)/x(24))-1;...      % 20. sRR
        sqrt(((x(1))^2 + (x(2))^2)) + x(8)*((tf/2) - a*x(6));...        % 21. vFL
        sqrt(((x(1))^2 + (x(2))^2)) + x(8)*((-tf/2) - a*x(6));...       % 22. vFR
        sqrt(((x(1))^2 + (x(2))^2)) + x(8)*((tr/2) + b*x(6));...        % 23. vRL
        sqrt(((x(1))^2 + (x(2))^2)) + x(8)*((-tr/2) + b*x(6));...       % 24. vRR
        -ckFL*x(17);...     % 25. fxFL
        -ckFR*x(18);...     % 26. fxFR
        -ckRL*x(19);...     % 27. fxRL
        -ckRR*x(20);...     % 28. fxRR
        -caFL*x(9);...      % 29. fyFL
        -caFR*x(10);...     % 30. fyFR
        -caRL*x(11);...     % 31. fyRL
        -caRR*x(12);...     % 32. fyRR
        x(33)+x(1)*dt;...   % 33. x
        x(34)+x(2)*dt;...   % 34. y
        x(35)+x(8)*dt];     % 35. yaw

%       x
%       meas
%       P
%       y_handle
%       Q_proc
%       R_meas
      [x, P] = ekf(x_handle,x,P,y_handle,meas,Q_proc,R_meas);
      x_store(:,k-1) = x;                            % save estimate
  end

  pause(0.05); refresh;

  % output statistics
  num_matches = visualOdometryMonoMex('num_matches');
  num_inliers = visualOdometryMonoMex('num_inliers');
  disp(['Frame: ' num2str(frame)]);
%   disp(['Frame: ' num2str(frame) ...
%         ', Matches: ' num2str(num_matches) ...
%         ', Inliers: ' num2str(100*num_inliers/num_matches,'%.1f') ,' %']);
end


figure;
  for k=1:3                                % plot results
    subplot(3,1,k)
    hold on;
%     plot(1:NN, meas_store(k,:), '-', 1:NN, x_store(k,:), '--')
%     plot(1:NN, gt(k,:), 'b-')
      plot(1:NN, gt(k,:), 'b-',1:NN, x_store(2*k-1,:), 'r--');
%     plot(1:NN, x_store(k,:), '--')
  end
  
%   figure;
%   for k=1:3                                % plot results
%     subplot(3,1,k)
%     hold on;
%     plot(1:NN, meas_store(k,:), 'b-')
%   end
%   
% % figure;
% %   for k=1:3                      
% %     subplot(3,1,k)
% %     plot(1:NN, radtodeg(eulAng(k,:)), '-')
% %   end
% % release visual odometry
% 
% figure;
% plot(gt(2,:)-gt(2,1), gt(1,:)-gt(1,1), 'b')
% axis equal
% figure;
% plot(X03(1:200,4),X03(1:200,12), 'b')
% axis equal
visualOdometryMonoMex('close');
