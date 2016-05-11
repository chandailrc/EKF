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
    n_states=13;
    n_meas_states = 9;                          % x, y, theta
    q_proc=0.1;                                 %std of process 
    r_meas=0.1;                                 %std of measurement
    
    Q_proc=q_proc^2*eye(n_states);              % covariance of process
    [Q_proc_E, Q_proc_e] = eig(Q_proc);
    R_meas=r_meas^2*eye(n_meas_states);         % covariance of measurement  
    [R_meas_E, R_meas_e] = eig(R_meas);
    dt = 0.1;
    delta = 0;
    WB = 2.71;
        % >>>>>>>>>>>>>NOISE START<<<<<<<<<<<<<<<<<<<<<
        mu = zeros(3,1);
        QR = [r_meas^2 0 0; 0 r_meas^2 0; 0 0 (r_meas*0.1)^2]; % Covariance
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
%     gt = [x z yaw roll pitch y]
    % Trans roll == pitch
    % Trans pitch == yaw
    % Trans yaw == roll
    
%     gt_meas = [gt(1,:)+0.1*randn(NN,1)'; gt(2,:)+0.1*randn(NN,1)'; gt(3,:)+0.01*randn(NN,1)'];
    gt_m = [gt(1,:); gt(2,:); gt(3,:)];
    gt_meas = gt_m + q;
    


    s_state=[gt(1,1) ;0;gt(1,2);0;gt(1,3);0;gt(1,4);0;gt(1,5);0;gt(1,6);0;0];         % initial state
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
      
      e = Q_proc_E*sqrt(Q_proc_e)*randn(n_states,1);
      d = R_meas_E*sqrt(R_meas_e)*randn(n_meas_states,1);
      
      
      % x_handle = [x vx z vz yaw yawR roll rollR pitch pitchR y vy V]
      % x_handle=@(x)[x(1)+x(2)*dt;x(13)*cos(x(5));x(3)+x(4)*dt;x(13)*sin(x(5));x(5)+x(6)*dt;x(6); x(7)+x(8)*dt; x(8); x(9)+x(10)*dt; x(10);x(11)+x(12)*dt;x(12); x(13)];
      x_handle=@(x)[x(1)+(x(2)*dt)+e(1);x(2)+e(2);x(3)+(x(4)*dt)+e(3);x(4)+e(4);x(5)+(x(6)*dt)+e(5);x(6)+e(6); x(7)+x(8)*dt+e(7); x(8)+e(8); x(9)+(x(10)*dt)+e(9); x(10)+e(10);x(11)+(x(12)*dt)+e(11);x(12)+e(12); x(13)+e(13)];
    
      % y_handle=@(x)[x(2)/dt;x(4)/dt;x(6)/dt];      % measurement equation
      y_handle=@(x)[x(1)+d(1);x(3)+d(2);x(5)+d(3);x(7)+d(4);x(9)+d(5);x(11)+d(6);x(1)+d(7);x(3)+d(8);x(5)+d(9)];
%       x
%       meas
%       P
%       y_handle
%       Q_proc
%       R_meas
      [x, P] = ekf(x_handle,x,P,y_handle,meas,Q_proc,R_meas);
      Rekf = eul2rotm([x(9) x(5) x(7)]);
%       disp([x(9) x(5) x(7)]);
      Tekf = [x(1);x(11);x(3)];
      Tr_ekf_total{k} = [Rekf Tekf; 0 0 0 1];
%       disp(rotm2eul(Tr_ekf_total{k}(1:3,1:3)));
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
 
  figure;
  for k=2:NN
  plot([Tr_total{k-1}(1,4) Tr_total{k}(1,4)], ...
         [Tr_total{k-1}(3,4) Tr_total{k}(3,4)],'-xb','LineWidth',1)
     hold on;
  end
  plot(gt(1,1:NN), gt(2,1:NN), 'r--', 'LineWidth',2);
  axis equal;
  hold on;
  for k=2:NN-1
      plot([x_store(1,k-1) x_store(1,k)], ...
           [x_store(3,k-1) x_store(3,k)],'-g+','LineWidth',1)
      hold on;
      theta = x_store(5,k);
      r = 1; % magnitude (length) of arrow to plot
      x = x_store(1,k); y = x_store(3,k);
      u = r * cos(theta); % convert polar (theta,r) to cartesian
      v = r * sin(theta);
      quiver(x,y,u,v);
      hold on;
  end
  hold off;
  
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

figure;
  for k=1:3                                % plot results
    subplot(3,1,k)
    hold on;
%     plot(1:NN, meas_store(k,:), '-', 1:NN, x_store(k,:), '--')
%     plot(1:NN, gt(k,:), 'b-')
      plot(1:NN, gt_m(k,:), 'b-',1:NN, gt_meas(k,:), 'r--');
%     plot(1:NN, x_store(k,:), '--')
  end
visualOdometryMonoMex('close');
