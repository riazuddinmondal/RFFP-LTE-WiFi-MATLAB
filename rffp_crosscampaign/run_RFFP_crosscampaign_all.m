%% Run Compact MLP first; then FastKAN, one direction at a time.
% Add the original DNN/RFFP helper directory to your MATLAB path first.
dataFile='RFFP_corrected_representations.mat';
% Compact MLP, both directions (checkpoint after each candidate).
mlpReports=run_RFFP_compact_MLP_crosscampaign(dataFile, ...
 'CompactMLP_crosscampaign_results',[1 2]);
% FastKAN: uncomment one direction at a time; default V2 settings.
% k1=run_RFFP_Hierarchical_FastKAN_crosscampaign(dataFile,1, ...
%   'OutputDir','KAN_crosscampaign_results','ExecutionEnvironment','auto');
% k2=run_RFFP_Hierarchical_FastKAN_crosscampaign(dataFile,2, ...
%   'OutputDir','KAN_crosscampaign_results','ExecutionEnvironment','auto');
