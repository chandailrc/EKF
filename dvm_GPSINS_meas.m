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
last_frame   = 200;

% init visual odometry
visualOdometryMonoMex('init',param);

% init transformation matrix array
Tr_total{1} = eye(4);
Tr_ekf_total{1} = eye(4);
Tr_ekf{1} = eye(4);

% create figure
figure('Color',[1 1 1]);
ha1 = axes('Position',[0.05,0.7,0.9,0.25]);
axis off;
ha2 = axes('Position',[0.05,0.05,0.9,0.6]);
set(gca,'XTick',-500:10:500);
set(gca,'ZTick',-500:10:500);
axis equal, grid on, hold on;


% for all frames do
replace = 0;

% >>>>>>>>START: EKF PARAMETERS<<<<<<<<<<<<<
    NN = last_frame - first_frame + 1;
    n_states=46;
    n_meas_states = 9;
%     n_meas_states_dash = 3;
    q_proc=0.1;                                 %std of process 
    r_meas= 0.1;                                 %std of measurement
    
    Q_proc=q_proc^2*eye(n_states);              % covariance of process
    Q_proc(5,5) = 0.01^2;
    Q_proc(6,6) = 0.01^2;
    Q_proc(7,7) = 0.01^2;
    Q_proc(8,8) = 0.01^2;
    Q_proc(9,9) = 0.01^2;
    Q_proc(10,10) = 0.01^2;
    [Q_proc_E, Q_proc_e] = eig(Q_proc);
    R_meas=r_meas^2*eye(n_meas_states);         % covariance of measurement
%     R_meas_dash=r_meas^2*eye(n_meas_states_dash);
    [R_meas_E, R_meas_e] = eig(R_meas);
%     [R_meas_E_dash, R_meas_e_dash] = eig(R_meas_dash);
    dt = 0.1;
    wb = 2.71;
    m = 1454;
    g = 9.81;
    tr = 1.6;
    tf = tr;
    a = 1.68;
    b = 2.71-1.68;
    h = 0.825;
    wheelr = (0.4/2)+0.11;
    
        % >>>>>>>>>>>>>NOISE START<<<<<<<<<<<<<<<<<<<<<
        mu = zeros(3,1);
        QR = [0.1^2 0 0; 0 0.1^2 0; 0 0 (0.01*0.01)^2]; % Covariance
        n = length(QR(:,1)); % Size
        [QRE, QRe] = eig(QR); % Eigenvectors and eigenvalues of Sigma

        % Create sample sets to represent the Gaussian distributions
        S =  NN;
        for i = 1:S
            ra(:,i) = randn(n,1);  % Generate normal samples
            q(:,i) = mu + QRE*sqrt(QRe)*ra(:,i); % Convert to samples with mean mu and covariance Q
        end

        % >>>>>>>>>>>>>NOISE END<<<<<<<<<<<<<<<<<<<<<<<
    load('D:\VO\data_odometry_gray\dataset\sequences\03\03.txt');
    gtdata = X03';
    for i=1:NN
       gtEulAng(:,i) = rotm2eul([X03(i,1:3); X03(i,5:7); X03(i,9:11)]);
    end
    gt = [gtdata(4,1:NN);gtdata(12,1:NN); gtEulAng(2,1:NN); gtEulAng(3,1:NN); gtEulAng(1,1:NN); gtdata(8,1:NN)];
    for i = 2:NN
        delta(i-1) = gtEulAng(2,i)-gtEulAng(2,i-1);
    end
    delta(201) = 0;
    velX(1) = 0;
    velY(1) = 0;
    for i=2:NN
        velX(i) = (gt(1,i)-gt(1,i-1))/dt;
        velY(i)  = (gt(2,i)-gt(2,i-1))/dt;
    end 
%     gt = [x z yaw roll pitch y]
    % Trans roll == pitch
    % Trans pitch == yaw
    % Trans yaw == roll
    
%     gt_meas = [gt(1,:)+0.1*randn(NN,1)'; gt(2,:)+0.1*randn(NN,1)'; gt(3,:)+0.01*randn(NN,1)'];
    gt_m = [gt(1,:); gt(2,:); gt(3,:)];
    gt_meas = gt_m + q;
    

% x_handle = [x vx z vz yaw yawR roll rollR pitch pitchR y vy Vxz ax az ay Axz fxFL fxFR fxRL fxRR fyFL fyFR fyRL fyRR]
    s_state=[...
        gt(1,1);...         % 1. x
        velX(2);...         % 2. vx
        gt(1,2);...         % 3. z
        10;...              % 4. vz
        gt(1,3);...         % 5. yaw
        1;...               % 6. yawR
        gt(1,4);...         % 7. roll
        1;...               % 8. rollR
        gt(1,5);...         % 9. pitch
        1;...               % 10. pitchR
        gt(1,6);...         % 11. y
        1;...               % 12. vy
        1;...               % 13. Vxz
        1;...               % 14. ax
        1;...               % 15. az
        1;...               % 16. ay
        1;...               % 17. Axz
        1;...               % 18. fxFL
        1;...               % 19. fxFR
        1;...               % 20. fxRL
        1;...               % 21. fxRR
        1;...               % 22. fyFL
        1;...               % 23. fyFR
        1;...               % 24. fyRL
        1;...               % 25. fyRR
        1;...               % 26. fzFL
        1;...               % 27. fzFR
        1;...               % 28. fzRL
        1;...               % 29. fzRR
        1;...               % 30. sFL
        1;...               % 31. sFR
        1;...               % 32. sRL
        1;...               % 33. sRR
        1;...               % 34. wFL
        1;...               % 35. wFR
        1;...               % 36. wRL
        1;...               % 37. wRR
        1;...               % 38. vFL
        1;...               % 39. vFR
        1;...               % 40. vRL
        1;...               % 41. vRR
        1;...               % 42. beta
        1;...               % 43. alphaFL
        1;...               % 44. alphaFR
        1;...               % 45. alphaRL
        1;...               % 46. alphaRR
        ];                  % initial state
    x=s_state+q_proc*randn(n_states,1);               % initial state with noise
%     x_dash = x;
    P = 0.1*eye(n_states);                            % initial state covraiance
%     P_dash = P;
    x_store = zeros(n_states,NN);                     %estmate    
    meas_store = zeros(n_meas_states,NN);             %actual
%     meas_store_dash = zeros(n_meas_states_dash,NN);
  
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
      Tr_ekf_total{k} = Tr_ekf_total{k-1};
      eulAng(:,k) = rotm2eul(Tr_total{k}(1:3,1:3));
      eulAngEkf(:,k) = rotm2eul(Tr_ekf_total{k}(1:3,1:3));
      
    % on success: update total motion (=pose)
    else
      replace = 0;
      Tr_total{k} = Tr_total{k-1}*inv(Tr);
      Tr_ekf_total{k} = Tr_ekf_total{k-1}*inv(Tr);
      eulAng(:,k) = rotm2eul(Tr_total{k}(1:3,1:3));
      eulAngEkf(:,k) = rotm2eul(Tr_ekf_total{k}(1:3,1:3));
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
    hold on;
    plot([Tr_ekf_total{k-1}(1,4) Tr_ekf_total{k}(1,4)], ...
         [Tr_ekf_total{k-1}(3,4) Tr_ekf_total{k}(3,4)],'-+r','LineWidth',1);

      
      
%       meas = [Tr_total{k}(1,4); Tr_total{k}(3,4);eulAng(2,k)];%; gt_meas(1,k);gt_meas(2,k);gt_meas(3,k)];
      meas = [Tr_ekf_total{k}(1,4); Tr_ekf_total{k}(3,4);eulAngEkf(2,k);eulAngEkf(3,k);eulAngEkf(1,k);Tr_ekf_total{k}(2,4); gt_meas(1,k);gt_meas(2,k);gt_meas(3,k)];
      meas_store(:,k)= meas;                            % save actual state 
      
%       meas_dash = [gt_meas(1,k);gt_meas(2,k);gt_meas(3,k)];
%       meas_store_dash(:,k)= meas_dash; 
      
      e = Q_proc_E*sqrt(Q_proc_e)*randn(n_states,1);
      d = R_meas_E*sqrt(R_meas_e)*randn(n_meas_states,1);
      
      
      % x_handle = [x vx z vz yaw yawR roll rollR pitch pitchR y vy Vxz ax az ay Axz fxFL fxFR fxRL fxRR fyFL fyFR fyRL fyRR]
%       x_handle=@(x)[x(1)+(x(2)*dt)+e(1);(x(13)*sin(x(5)))+e(2);x(3)+(x(4)*dt)+e(3);(x(13)*cos(x(5)))+e(4);x(5)+(x(6)*dt)+e(5);x(6)+e(6); x(7)+(x(8)*dt)+e(7); x(8)+e(8); x(9)+(x(10)*dt)+e(9); x(10)+e(10);x(11)+(x(12)*dt)+e(11);x(12)+e(12); x(13)+e(13)];
      x_handle=@(x)[...
          x(1)+(x(2)*dt+(0.5*x(14)*dt*dt))+e(1);...                 % 1. x
          x(2)+(x(14)*dt)+e(2);...                                  % 2. vx
          x(3)+(x(4)*dt+(0.5*x(15)*dt*dt))+e(3);...                 % 3. z
          x(4)+(x(15)*dt)+e(4);...                                  % 4. vz
          x(5)+(x(6)*dt)+e(5);...                                   % 5. yaw
          sqrt(((x(1))^2 + (x(2))^2))*tan(delta(k))/wb;...          % 6. yawR
          x(7)+(x(8)*dt)+e(7);...                                   % 7. roll
          x(8)+e(8);...                                             % 8. rollR
          x(9)+(x(10)*dt)+e(9);...                                  % 9. pitch
          x(10)+e(10);...                                           % 10. pitchR
          x(11)+(x(12)*dt)+e(11);...                                % 11. y
          x(12)+e(12);...                                           % 12. vy
          x(13)+(x(17)*dt)+e(13);...                                % 13. Vxz
          (((1/m)*((x(18)*cos(delta(k)))-(x(22)*sin(delta(k)))+x(19)*cos(delta(k))-x(23)*sin(delta(k))+x(20)+x(21)))*sin(x(5)))+e(14);...           % 14. ax
          (((1/m)*((x(22)*cos(delta(k)))+(x(18)*sin(delta(k)))+x(23)*cos(delta(k))+x(19)*sin(delta(k))+x(24)+x(25)))*cos(x(5)))+e(15);...           % 15. az
          x(16)+e(16);...                                           % 16. ay
          x(17)+e(17);...                                           % 17. Axz
          x(18)+e(18);...                                           % 18. fxFL
          x(19)+e(19);...                                           % 19. fxFR
          x(20)+e(20);...                                           % 20. fxRL
          x(21)+e(21);...                                           % 21. fxRR
          x(22)+e(22);...                                           % 22. fyFL
          x(23)+e(23);...                                           % 23. fyFR
          x(24)+e(24);...                                           % 24. fyRL
          x(25)+e(25);...                                           % 25. fyRR
          (0.5*m*g + (m*x(15)*h)/tf)*(b/(a+b)) - 0.5*m*x(14)*(h/a+b) + e(26);...        % 26. fzFL
          (0.5*m*g - (m*x(15)*h)/tf)*(b/(a+b)) - 0.5*m*x(14)*(h/a+b) + e(27);...        % 27. fzFR
          (0.5*m*g + (m*x(15)*h)/tr)*(b/(a+b)) + 0.5*m*x(14)*(h/a+b) + e(28);...        % 28. fzRL
          (0.5*m*g - (m*x(15)*h)/tr)*(b/(a+b)) + 0.5*m*x(14)*(h/a+b) + e(29);...        % 29. fzRR
          ((x(34)*wheelr)/x(38))-1 + e(30);...      % 30. sFL
          ((x(35)*wheelr)/x(39))-1 + e(31);...      % 31. sFR
          ((x(36)*wheelr)/x(40))-1 + e(32);...      % 32. sRL
          ((x(37)*wheelr)/x(41))-1 + e(33);...      % 33. sRR
          x(34);...                         % 34. wFL
          x(35);...                         % 35. wFR
          x(36);...                         % 36. wRL
          x(37);...                         % 37. wRR
          sqrt(((x(2))^2 + (x(4))^2)) + x(6)*((tf/2) - a*x(42));...        % 38. vFL
          sqrt(((x(2))^2 + (x(4))^2)) + x(6)*((-tf/2) - a*x(42));...       % 39. vFR
          sqrt(((x(2))^2 + (x(4))^2)) + x(6)*((tr/2) + b*x(42));...        % 40. vRL
          sqrt(((x(2))^2 + (x(4))^2)) + x(6)*((-tr/2) + b*x(42));...       % 41. vRR
          atan((x(4)*cos(x(5)))/(x(2)*sin(x(5))));...               % 42. beta
          delta(k)-atan((x(4)+(a*x(6)))/(x(1)+(0.5*tf*x(6))));...     % 43. alphaFL
          delta(k)-atan((x(4)+(a*x(6)))/(x(1)-(0.5*tf*x(6))));...     % 44. alphaFR
          atan((-x(4)+b*x(6))/(x(1)+tr*(x(6)/2)));...          % 45. alphaRL
          atan((-x(4)+b*x(6))/(x(1)-tr*(x(6)/2)));...          % 46. alphaRR
          ];
%       x_handle=@(x)[x(1)+(x(2)*dt)+e(1);x(2)+e(2);x(3)+(x(4)*dt)+e(3);x(4)+e(4);x(5)+(x(6)*dt)+e(5);x(6)+e(6); x(7)+x(8)*dt+e(7); x(8)+e(8); x(9)+(x(10)*dt)+e(9); x(10)+e(10);x(11)+(x(12)*dt)+e(11);x(12)+e(12); x(13)+e(13)];
%       x_handle_dash=@(x)[x(1)+(x(2)*dt)+e(1);x(2)+e(2);x(3)+(x(4)*dt)+e(3);x(4)+e(4);x(5)+(x(6)*dt)+e(5);x(6)+e(6); x(7)+x(8)*dt+e(7); x(8)+e(8); x(9)+(x(10)*dt)+e(9); x(10)+e(10);x(11)+(x(12)*dt)+e(11);x(12)+e(12); x(13)+e(13)];
      

%       y_handle=@(x)[x(2)/dt;x(4)/dt;x(6)/dt];      % measurement equation
      y_handle=@(x)[x(1)+d(1);x(3)+d(2);x(5)+d(3);x(7)+d(4);x(9)+d(5);x(11)+d(6);x(1)+d(7);x(3)+d(8);x(5)+d(9)];
      
%       y_handle_dash=@(x)[x(1)+d(7);x(3)+d(8);x(5)+d(9)];
%       x
%       meas
%       P
%       y_handle
%       Q_proc
%       R_meas
      [x, P] = ekf(x_handle,x,P,y_handle,meas,Q_proc,R_meas);
%       [x_dash, P_dash] = ekf(x_handle_dash,x_dash,P_dash,y_handle_dash,meas_dash,Q_proc,R_meas_dash);
      Rekf = eul2rotm([x(9) x(5) x(7)]);
%       disp([x(9) x(5) x(7)]);
      Tekf = [x(1);x(11);x(3)];
      Tr_ekf_total{k} = [Rekf Tekf; 0 0 0 1];
%       disp(rotm2eul(Tr_ekf_total{k}(1:3,1:3)));
      x_store(:,k-1) = x;                            % save estimate
%       x_store_dash(:,k-1) = x_dash; 
        
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


% figure;
%   for k=1:3                                % plot results
%     subplot(3,1,k)
%     hold on;
% %     plot(1:NN, meas_store(k,:), '-', 1:NN, x_store(k,:), '--')
% %     plot(1:NN, gt(k,:), 'b-')
%       plot(1:NN, gt(k,:), 'b-',1:NN, x_store(2*k-1,:), 'r--', 1:NN, gt_meas(k,:), 'g--');
% %     plot(1:NN, x_store(k,:), '--')
%   end
 
%   figure;
%   for k=2:NN
%   plot([Tr_total{k-1}(1,4) Tr_total{k}(1,4)], ...
%          [Tr_total{k-1}(3,4) Tr_total{k}(3,4)],'-xb','LineWidth',1)
%      hold on;
%   end
%   plot(gt(1,1:NN), gt(2,1:NN), 'r--', 'LineWidth',2);
%   axis equal;
%   hold on;

% figure;
%   for k=2:NN-1
% %       plot([gt(1,k+1-1) gt(1,k+1)],...
% %            [gt(2,k+1-1) gt(2,k+1)],'-ro','LineWidth',1)
% %       hold on;
%       plot([x_store(1,k-1) x_store(1,k)], ...
%            [x_store(3,k-1) x_store(3,k)],'-g+','LineWidth',1)
%       hold on;
%       theta = x_store(5,k);
%       r = 1; % magnitude (length) of arrow to plot
%       x = x_store(1,k); y = x_store(3,k);
%       u = r * cos(theta); % convert polar (theta,r) to cartesian
%       v = r * sin(theta);
%       quiver(x,y,u,v);
%       hold on;
%       plot([gt_meas(1,k+1-1) gt_meas(1,k+1)],...
%            [gt_meas(2,k+1-1) gt_meas(2,k+1)],'-bx','LineWidth',1)
%       hold on;
%   end
%   xlabel('X');
%   ylabel('Y');
%   hold off;
  
  figure
  for k=1:3                                % plot results
    subplot(5,1,k)
    plot(1:NN, gt(k,:), 'b-', 1:NN, x_store(2*k-1,:), 'r--')
  end
  subplot(5,1,4)
  plot(1:NN, velX, 'b-', 1:NN, x_store(2,:),'r--');
  subplot(5,1,5)
  plot(1:NN, velY, 'b-', 1:NN, x_store(4,:),'r--');
  for k=2:NN-1
      estDevX(k-1) = abs((x_store(1,k-1)-gt(1,k+1-1)));
      estDevY(k-1) = abs((x_store(3,k-1)-gt(2,k+1-1)));
      estDevVX(k-1) = abs((x_store(2, k-1)-velX(k)));
      estDevVY(k-1) = abs((x_store(4, k-1)-velY(k)));
%       estDevX_dash(k-1) = abs((x_store_dash(1,k-1)-gt(1,k+1-1))/gt(1,k+1-1));
%       estDevY_dash(k-1) = abs((x_store_dash(3,k-1)-gt(2,k+1-1))/gt(2,k+1-1));
      measDevX(k-1) = abs((gt_meas(1,k+1-1)-gt(1,k+1-1)));
      measDevY(k-1) = abs((gt_meas(2,k+1-1)-gt(2,k+1-1)));
  end
  
  % Relative Percent Difference
  
  
  mEDX = mean(estDevX);
  mEDY = mean(estDevY);
  mEDVX = mean(estDevVX);
  mEDVY = mean(estDevVY);
%   mEDX_dash = mean(estDevX_dash);
%   mEDY_dash = mean(estDevY_dash);
  mMDX = mean(measDevX);
  mMDY = mean(measDevY);
  max(estDevVX);
  max(estDevVY);
  
  %{
  figure;
   for k=2:NN-1
%       plot([gt(1,k+1-1) gt(1,k+1)],...
%            [gt(2,k+1-1) gt(2,k+1)],'-ro','LineWidth',1)
%       hold on;
      plot([x_store_dash(1,k-1) x_store_dash(1,k)], ...
           [x_store_dash(3,k-1) x_store_dash(3,k)],'-g+','LineWidth',1)
      hold on;
      
      plot([x_store(1,k-1) x_store(1,k)], ...
           [x_store(3,k-1) x_store(3,k)],'-r+','LineWidth',1)
      hold on;
%       theta = x_store(5,k);
%       r = 1; % magnitude (length) of arrow to plot
%       x = x_store(1,k); y = x_store(3,k);
%       u = r * cos(theta); % convert polar (theta,r) to cartesian
%       v = r * sin(theta);
%       quiver(x,y,u,v);
%       hold on;
      plot([gt_meas(1,k+1-1) gt_meas(1,k+1)],...
           [gt_meas(2,k+1-1) gt_meas(2,k+1)],'-bx','LineWidth',1)
      hold on;
  end
  xlabel('X');
ylabel('Y');
  hold off;
%}
%   figure;
%   plot(1:NN, gt_meas(3,:),'b-',1:NN, x_store(5,:), 'r--');
      
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

% figure;
%   for k=1:3                                % plot results
%     subplot(3,1,k)
%     hold on;
% %     plot(1:NN, meas_store(k,:), '-', 1:NN, x_store(k,:), '--')
% %     plot(1:NN, gt(k,:), 'b-')
%       plot(1:NN, gt_m(k,:), 'b-',1:NN, gt_meas(k,:), 'r--');
% %     plot(1:NN, x_store(k,:), '--')
%   end
visualOdometryMonoMex('close');
