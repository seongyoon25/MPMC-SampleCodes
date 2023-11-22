%{
    File Name: bayesian_parameter_identification.m
    Author: Seongyoon Kim (seongyoonk25@gmail.com)
    Date: 2023-05-16
    Description:
    This script demonstrates how to use UQLab to perform a Bayesian parameter
    identification for a given model. The main steps are as follows:
    1. Initialization and adding UQ Lab to path
    2. Setting reference parameters and getting reference model outputs
    3. UQ Lab input setting (prior distribution)
    4. UQ Lab forward model setting
    5. UQ Lab observation setting
    6. UQ Lab discrepancy setting (model/observation error)
    7. UQ Lab solver setting (MCMC sampler)
    8. UQ Lab inversion setting
    9. Running UQ Lab Bayesian parameter identification
   10. Printing and plotting the results
   11. Post-processing: removing bad chains and getting maximum a posteriori (MAP)
   12. Post-processing: getting samples and corresponding model outputs
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

%% Get reference model outputs
% If the model's input goes into NxP, the model's output must come out of NxQ, 
% where P and Q are the dimensionality of the input parameters and the output.
otherArg = ...
outRef = model(parRef, otherArg);

%% UQ Lab input setting
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

%% UQ Lab forward model setting
modelOpts.Name = 'Good model'; 
modelOpts.mHandle = @(par) model(par, otherArg);
modelOpts.isVectorized = true;  % true for multivariate output

myForwardModel = uq_createModel(modelOpts);

%% UQ Lab observation setting
myData.Name = 'Reference value';
myData.y = outRef;

%% UQ Lab discrepancy setting
% Case 1: model/observation error with fixed variance 
discrepancyOpts.Type = 'Gaussian';
discrepancyOpts.Parameters = 0.005^2;

% Case 2: model/observation error with variance estimation
% discrepancyPriorOpts.Marginals.Type = 'Lognormal';
% discrepancyPriorOpts.Marginals.Parameters = [0 1];
% myDiscrepancyPrior = uq_createInput(discrepancyPriorOpts);
% discrepancyOpts.Type = 'Gaussian';
% discrepancyOpts.Prior = myDiscrepancyPrior;

bayesOpts.Discrepancy = discrepancyOpts;

%% UQ Lab solver setting
solver.Type = 'MCMC';
solver.MCMC.Sampler = 'AIES';  % 'AIES' for default
solver.MCMC.NChains = 100;  % 100 for default
solver.MCMC.Steps = 1e3;  % 300 for default

solver.MCMC.Visualize.Parameters = [1 3 5];  % select parameters for chain visualization
solver.MCMC.Visualize.Interval = 10;  % how often to figure update

bayesOpts.Solver = solver;

%% UQ Lab inversion setting
bayesOpts.Type = 'Inversion';
bayesOpts.Data = myData;
bayesOpts.Prior = myPriorDist;

%% Start UQ Lab Bayesian parameter identification
tic
myBayesianAnalysis = uq_createAnalysis(bayesOpts);
toc

%% Print results
uq_print(myBayesianAnalysis)

%% Plot results
% uq_display(myBayesianAnalysis)
uq_display(myBayesianAnalysis, 'scatterplot', 1:length(parNames))  % parameter distribution
uq_display(myBayesianAnalysis, 'acceptance', true)  % acceptance rate for each chain
uq_display(myBayesianAnalysis, 'predDist', true)  % model output distribution

%% Post-processing: remove bad chains
uq_display(myBayesianAnalysis, 'acceptance', true)  % acceptance rate for each chain
acceptance = myBayesianAnalysis.Results.Acceptance;
[~,tolL,tolU,tolC] = isoutlier(acceptance, 'ThresholdFactor', 2);  % get 2sigma tolerance
TF = acceptance < tolC;  % disregard lower half chains
badChains = find(TF);
yline(tolL,'b--');
yline(tolU,'b--');
yline(tolC,'b-.');
uq_postProcessInversion(myBayesianAnalysis,'badChains',badChains);
hold on
scatter(badChains, acceptance(badChains), 'red', 'filled')
legend('All chains', '', '', '', 'Bad chains')

%% Post-processing: get maximum a posteriori (MAP)
uq_postProcessInversion(myBayesianAnalysis,'pointEstimate','MAP');
parMap = myBayesianAnalysis.Results.PostProc.PointEstimate.X{1,1};  % this changes from mean to map
outMap = model(parMap, otherArg);

%% Post-processing: get samples and corresponding model outputs
samplePar = myBayesianAnalysis.Results.PostProc.PostSample;  % NSteps x P x NChains
sampleOutput = myBayesianAnalysis.Results.PostProc.PostModel.evaluation;  % NSteps x Q x NChains