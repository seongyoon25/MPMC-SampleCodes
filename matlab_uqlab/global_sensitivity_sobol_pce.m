%{
    File Name: global_sensitivity_sobol_pce.m
    Author: Seongyoon Kim (seongyoonk25@gmail.com)
    Date: 2023-05-16
    Description:
    This script demonstrates how to use UQLab to perform 
    a polynomial chaos expansion (PCE)-based global sensitivity analysis.
    It includes the following steps:
    1. Initialization and adding UQ Lab to path
    2. Setting reference parameters
    3. UQ Lab input setting
    4. UQ Lab polynomial chaos expansion (PCE) setting
    5. UQ Lab global sensitivity analysis (GSA) setting and run
    6. Plotting the results
%}

%% Initialization
clear all;
close all;
warning off;
clc;

%% Add UQ Lab and other function files to path
addpath(genpath(pwd), '-end');
addpath(genpath('path/to/uqlab/UQLab_Rel2.0.0/'), '-end');
uqlab;

%% Set reference parameters
parNames = ["aaa", "bbb", ...
            "ccc", "ddd"];

parRef(:,1) = ...
parRef(:,2) = ...
parRef(:,3) = ...
parRef(:,4) = ...

%% UQ Lab input setting
% Since the polynomial chaos expansion (PCE) will be configured based on the sample, 
% `priorOpts` are just being written formally for the uqlab run.
r = .4;
priorType = 'Lognormal';  % Set lognormal prior distribution for positive parameters
priorMoments = [1 r];
priorOpts.Name = 'Model parameters prior';

for i = 1:length(parNames)
    priorOpts.Marginals(i).Name = parNames(i);
    priorOpts.Marginals(i).Type = priorType;
    priorOpts.Marginals(i).Moments = priorMoments * parRef(i);  % mean and standard deviation for Moments
end

myPriorDist = uq_createInput(priorOpts);

%% UQ Lab polynomial chaos expansion (PCE) setting
metaOpts.Type = 'Metamodel';
metaOpts.MetaType = 'PCE';

% Case 1: from any sample
metaOpts.ExpDesign.X = ... %  must be NxP, where N is sample number
metaOpts.ExpDesign.Y = ... %  must be NxQ, where N is sample number

% Case 2: from Bayesian parameter identification
% dimParameter = ...
% dimOutput = ...
% metaOpts.ExpDesign.X = reshape(permute(myBayesianAnalysis.Results.PostProc.PostSample, [1,3,2]), [], dimParameter);
% metaOpts.ExpDesign.Y = reshape(permute(myBayesianAnalysis.Results.PostProc.PostModel.evaluation, [1,3,2]), [], dimOutput);

metaOpts.TruncOptions.qNorm = 0.75;
metaOpts.Degree = 3;
metaOpts.Method = 'OLS'; % 'LARS' or 'OLS'

%% Start UQ Lab PCE and save
myPCE = uq_createModel(metaOpts);
save('result_GSA_PCE.mat', 'myPCE')

%% UQ Lab global sensitivity analysis (GSA) setting and run
pceSobol.Type = 'Sensitivity';
pceSobol.Method = 'Sobol';
pceSobol.Sobol.Order = 3;
pceSobol.Model = myPCE;
pceSobolAnalysis = uq_createAnalysis(pceSobol);

%% Get averaged global sensitivity
totalSobol = mean(pceSobolAnalysis.Results.Total, 2);

%% Plot the results
figure
bar(totalSobol)
title('Average Sensitivity Index')
xticks(1:length(parNames))
xticklabels(parNames)