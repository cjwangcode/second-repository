% LoRA deployment simulation for collaborative LAE.

clear;
clc;
close all;
rng(1);

if exist('cvx_begin', 'file') ~= 2
    error('CVX is required. Please run cvx_setup before this script.');
end
cvx_precision medium;

p = make_simulation_parameters();

outDir = fullfile(pwd, 'output', 'matlab');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

% fprintf('LoRA deployment simulation with CVX\n');
% fprintf('CCS -> edge data: %.2f MB\n', p.dataCcsMb);
% fprintf('Edge -> UAV data: %.2f MB\n', p.dataEdgeMb);
% fprintf('Edge workload: %.3e FLOPs\n', p.fEdgeWork);
% fprintf('UAV workload: %.3e FLOPs\n', p.fUavWork);
% fprintf('CCS -> edge latency threshold: %.2f s\n', p.tCcsMax);
% fprintf('Deployment interval tau: %.2f s\n\n', p.tau);

hCcsEdge = build_ccs_edge_channels(p);
hEdgeUav = build_edge_uav_channels(p);

Wccs = solve_ccs_beamforming_cvx(hCcsEdge, p);
[rateCcs, latencyCcs] = ccs_rates_from_covariance(Wccs, hCcsEdge, p);
powerCcs = real(trace(Wccs));

if max(latencyCcs) > p.tCcsMax + p.feasTol
    error('CCS-to-edge latency is %.4f s, which violates the 10 s requirement.', max(latencyCcs));
end

beta = solve_exact_association(hEdgeUav, p);

pEdge = 0.85 * p.pEdgeMax * ones(p.G, 1);
bwRatio = ones(p.G, 1) / p.G;
tEdge = edge_transmission_times(beta, bwRatio, pEdge, hEdgeUav, p);

objHistory = nan(p.maxIters, 1);
ccsLatencyHistory = nan(p.maxIters, 1);
maxTotalLatencyHistory = nan(p.maxIters, 1);

alphaEdge = p.kEdge * ones(p.G, 1);
alphaUav = p.kUav * ones(p.U, 1);
[lUpdEdge, lUpdUav] = update_latencies_from_alpha(alphaEdge, alphaUav, p);

for iter = 1:p.maxIters
    [alphaEdgeOpt, alphaUavOpt] = solve_gpu_scheduling_cvx( ...
        latencyCcs, tEdge, beta, p);
    alphaEdge = p.relaxation * alphaEdge + (1 - p.relaxation) * alphaEdgeOpt;
    alphaUav = p.relaxation * alphaUav + (1 - p.relaxation) * alphaUavOpt;
    [lUpdEdge, lUpdUav] = update_latencies_from_alpha(alphaEdge, alphaUav, p);

    [bwRatioOpt, ~] = solve_bandwidth_allocation_cvx( ...
        pEdge, latencyCcs, lUpdEdge, lUpdUav, beta, hEdgeUav, p);

    bwRatio = p.relaxation * bwRatio + (1 - p.relaxation) * bwRatioOpt;
    bwRatio = bwRatio / sum(bwRatio);
    tEdge = edge_transmission_times(beta, bwRatio, pEdge, hEdgeUav, p);

    tBudget = edge_time_budgets(latencyCcs, lUpdEdge, lUpdUav, beta, p);
    tPowerTarget = p.relaxation * tEdge + (1 - p.relaxation) * tBudget;
    tPowerTarget = min(tPowerTarget, tBudget);
    pEdgeOpt = solve_edge_power_cvx(tPowerTarget, bwRatio, beta, hEdgeUav, p);
    pEdge = p.relaxation * pEdge + (1 - p.relaxation) * pEdgeOpt;
    tEdge = edge_transmission_times(beta, bwRatio, pEdge, hEdgeUav, p);

    [lUpdEdge, lUpdUav] = update_latencies_from_alpha(alphaEdge, alphaUav, p);

    [objHistory(iter), maxTotalLatencyHistory(iter)] = evaluate_objective( ...
        powerCcs, latencyCcs, pEdge, tEdge, alphaEdge, alphaUav, ...
        lUpdEdge, lUpdUav, beta, p);
    ccsLatencyHistory(iter) = max(latencyCcs);

    fprintf(['iter %02d: objective %.6e J, max total latency %.4f s, ', ...
        'max CCS-edge latency %.4f s\n'], ...
        iter, objHistory(iter), maxTotalLatencyHistory(iter), ccsLatencyHistory(iter));

    if iter >= 4
        relGap = abs(objHistory(iter) - objHistory(iter - 1)) / max(1, abs(objHistory(iter - 1)));
        if relGap <= p.stopTol
            objHistory = objHistory(1:iter);
            ccsLatencyHistory = ccsLatencyHistory(1:iter);
            maxTotalLatencyHistory = maxTotalLatencyHistory(1:iter);
            break;
        end
    end
end

[alphaEdgeInt, alphaUavInt] = integerize_gpu_allocation(alphaEdge, alphaUav, p);
[lUpdEdgeInt, lUpdUavInt] = update_latencies_from_alpha(alphaEdgeInt, alphaUavInt, p);
[objectiveInteger, maxTotalLatencyInteger] = evaluate_objective( ...
    powerCcs, latencyCcs, pEdge, tEdge, alphaEdgeInt, alphaUavInt, ...
    lUpdEdgeInt, lUpdUavInt, beta, p);

save(fullfile(outDir, 'lora_cvx_results.mat'), ...
    'p', 'Wccs', 'powerCcs', 'rateCcs', 'latencyCcs', 'beta', ...
    'bwRatio', 'pEdge', 'tEdge', 'alphaEdge', 'alphaUav', ...
    'alphaEdgeInt', 'alphaUavInt', 'lUpdEdge', 'lUpdUav', ...
    'lUpdEdgeInt', 'lUpdUavInt', 'objectiveInteger', ...
    'maxTotalLatencyInteger', 'objHistory', 'ccsLatencyHistory', ...
    'maxTotalLatencyHistory');
writematrix(beta, fullfile(outDir, 'beta_association.csv'));
writematrix([alphaEdgeInt(:).', alphaUavInt(:).'], fullfile(outDir, 'integer_gpu_allocation.csv'));

plot_convergence(objHistory, outDir);

fprintf('\nFinal summary\n');
fprintf('Relaxed objective: %.6e J\n', objHistory(end));
fprintf('Integer objective: %.6e J\n', objectiveInteger);
fprintf('CCS transmit power: %.4f W\n', powerCcs);
fprintf('Max CCS -> edge latency: %.4f s\n', max(latencyCcs));
fprintf('Max deployment latency with integer GPUs: %.4f s\n', maxTotalLatencyInteger);
fprintf('Edge bandwidth ratios: %s\n', mat2str(bwRatio.', 4));
fprintf('Edge transmit powers: %s W\n', mat2str(pEdge.', 4));
fprintf('Edge transmit times: %s s\n', mat2str(tEdge.', 4));
fprintf('Beta association matrix rows=edge servers, columns=UAVs:\n');
disp(beta);
fprintf('Edge GPU allocation: %s\n', mat2str(alphaEdgeInt.'));
fprintf('UAV GPU allocation: %s\n', mat2str(alphaUavInt.'));
fprintf('Convergence plot saved to: %s\n', fullfile(outDir, 'lora_objective_convergence.png'));

function p = make_simulation_parameters()
    p.G = 2;
    p.U = 4;
    p.M = 6;
    p.betaMax = 2;

    p.dataCcsMb = 671.22;
    p.dataEdgeMb = 190.44;
    p.bytesPerMb = 1e6;
    p.dataCcsBits = p.dataCcsMb * p.bytesPerMb * 8;
    p.dataEdgeBits = p.dataEdgeMb * p.bytesPerMb * 8;

    p.tCcsMax = 9.99;
    p.tau = 1000.0;

    p.B = 100e6;
    p.carrierHz = 2.4e9;
    p.lambda = 3e8 / p.carrierHz;
    p.antennaSpacing = p.lambda / 2;
    p.beta0 = (p.lambda / (4 * pi))^2;
    p.pathlossExpCcs = 2.2;
    p.pathlossExpEdge = 2.8;
    p.ricianCcs = 8;
    p.ricianEdge = 5;
    p.extraLossCcsDb = 0;
    p.extraLossEdgeDb = 20;
    p.noiseFigureDb = 7;
    p.noiseDensityWPerHz = 10^((-174 - 30) / 10);
    p.noisePower = p.noiseDensityWPerHz * p.B * 10^(p.noiseFigureDb / 10);

    p.pCcsMax = 30.0;
    p.pEdgeMax = 8.0;

    p.ccsPos = [0, 0, 80];
    p.edgeXY = [1000, -300;
                1200,  250;
                1450, -100];
    p.uavXY = [ 960, -360;
               1040, -240;
               1160,  190;
               1250,  310;
               1400, -150;
               1510,  -40];
    p.uavH = 120;

    p.alphaEdgeMin = 1;
    p.alphaUavMin = 2;
    p.kEdge = 8;
    p.kUav = 8;

    p.fEdge = 4.68e12;
    p.fUav = 8.90e12;

    p.fEdgeWork = 5.26e15;
    p.fUavWork = 8.7e15;

    p.xiEdge = 179.0;
    p.xiUav = 336.1;

    p.pGpuEdge = 250.0;
    p.pGpuUav = 120.0;

    p.maxIters = 40;
    p.relaxation = 0.62;
    p.stopTol = 1e-5;
    p.feasTol = 1e-6;
end

function h = build_ccs_edge_channels(p)
    h = zeros(p.M, p.G);
    for g = 1:p.G
        edgePos = [p.edgeXY(g, :), 0];
        delta = edgePos - p.ccsPos;
        dist = norm(delta);
        theta = acos(abs(delta(3)) / dist);
        phase = 2 * pi * p.antennaSpacing / p.lambda * cos(theta) * (0:p.M - 1).';
        aLos = exp(-1i * phase);
        aNlos = (randn(p.M, 1) + 1i * randn(p.M, 1)) / sqrt(2);
        pathloss = p.beta0 * dist^(-p.pathlossExpCcs) * 10^(-p.extraLossCcsDb / 10);
        h(:, g) = sqrt(pathloss) * ( ...
            sqrt(p.ricianCcs / (p.ricianCcs + 1)) * aLos + ...
            sqrt(1 / (p.ricianCcs + 1)) * aNlos);
    end
end

function h = build_edge_uav_channels(p)
    h = zeros(p.G, p.U);
    for g = 1:p.G
        edgePos = [p.edgeXY(g, :), 0];
        for u = 1:p.U
            uavPos = [p.uavXY(u, :), p.uavH];
            dist = norm(uavPos - edgePos);
            nlos = (randn + 1i * randn) / sqrt(2);
            pathloss = p.beta0 * dist^(-p.pathlossExpEdge) * 10^(-p.extraLossEdgeDb / 10);
            h(g, u) = sqrt(pathloss) * ( ...
                sqrt(p.ricianEdge / (p.ricianEdge + 1)) + ...
                sqrt(1 / (p.ricianEdge + 1)) * nlos);
        end
    end
end

function W = solve_ccs_beamforming_cvx(h, p)
    snrReq = 2^(p.dataCcsBits / (p.B * p.tCcsMax)) - 1;
    cvx_solver Mosek
    cvx_begin sdp quiet
        variable W(p.M, p.M) hermitian semidefinite
        minimize(real(trace(W)))
        subject to
            real(trace(W)) <= p.pCcsMax;
            for g = 1:p.G
                Hg = h(:, g) * h(:, g)';
                real(trace(Hg * W)) >= p.noisePower * snrReq;
            end
    cvx_end
    require_cvx_success(cvx_status, 'CCS beamforming');
end

function [rate, latency] = ccs_rates_from_covariance(W, h, p)
    rate = zeros(p.G, 1);
    for g = 1:p.G
        Hg = h(:, g) * h(:, g)';
        snr = real(trace(Hg * W)) / p.noisePower;
        rate(g) = p.B * log2(1 + max(snr, 0));
    end
    latency = p.dataCcsBits ./ rate;
end

function beta = solve_exact_association(hEdgeUav, p)
    numAssignments = p.G ^ p.U;
    bestScore = -inf;
    bestAssign = ones(1, p.U);
    gain = abs(hEdgeUav).^2;

    for id = 0:numAssignments - 1
        assign = zeros(1, p.U);
        x = id;
        for u = 1:p.U
            assign(u) = mod(x, p.G) + 1;
            x = floor(x / p.G);
        end
        counts = accumarray(assign.', 1, [p.G, 1]).';
        if any(counts > p.betaMax)
            continue;
        end
        selectedGain = zeros(1, p.U);
        for u = 1:p.U
            selectedGain(u) = gain(assign(u), u);
        end
        score = min(log(selectedGain + eps)) + 1e-3 * sum(log(selectedGain + eps));
        if score > bestScore
            bestScore = score;
            bestAssign = assign;
        end
    end

    beta = zeros(p.G, p.U);
    for u = 1:p.U
        beta(bestAssign(u), u) = 1;
    end
end

function [alphaEdge, alphaUav] = solve_gpu_scheduling_cvx( ...
    latencyCcs, tEdge, beta, p)
    cvx_begin quiet
        variables alphaEdge(p.G) alphaUav(p.U)
        edgeEnergyVariable = p.pGpuEdge * p.xiEdge * sum(alphaEdge);
        uavEnergyVariable = p.pGpuUav * p.xiUav * sum(alphaUav);
        minimize(edgeEnergyVariable + uavEnergyVariable)
        subject to
            p.alphaEdgeMin <= alphaEdge <= p.kEdge;
            p.alphaUavMin <= alphaUav <= p.kUav;
            for g = 1:p.G
                for u = find(beta(g, :) > 0)
                    latencyCcs(g) + ...
                        (p.fEdgeWork / p.fEdge) * inv_pos(alphaEdge(g)) + p.xiEdge + ...
                        tEdge(g) + ...
                        (p.fUavWork / p.fUav) * inv_pos(alphaUav(u)) + p.xiUav <= p.tau;
                end
            end
    cvx_end
    require_cvx_success(cvx_status, 'GPU scheduling');
end

function [lUpdEdge, lUpdUav] = update_latencies_from_alpha(alphaEdge, alphaUav, p)
    lUpdEdge = p.fEdgeWork ./ (alphaEdge * p.fEdge) + p.xiEdge;
    lUpdUav = p.fUavWork ./ (alphaUav * p.fUav) + p.xiUav;
end

function [alphaEdgeInt, alphaUavInt] = integerize_gpu_allocation(alphaEdge, alphaUav, p)
    alphaEdgeInt = min(p.kEdge, max(p.alphaEdgeMin, ceil(alphaEdge - 1e-8)));
    alphaUavInt = min(p.kUav, max(p.alphaUavMin, ceil(alphaUav - 1e-8)));
end

function [bwRatio, tEdge] = solve_bandwidth_allocation_cvx( ...
    pEdge, latencyCcs, lUpdEdge, lUpdUav, beta, hEdgeUav, p)
    linkRateNoGamma = p.B * log2(1 + link_snr(pEdge, hEdgeUav, p));
    minTimeAtFullBandwidth = p.dataEdgeBits ./ linkRateNoGamma;
    cvx_solver SDPT3
    cvx_begin quiet
        variables bwRatio(p.G) tEdge(p.G)
        minimize(pEdge(:)' * tEdge)
        subject to
            bwRatio >= 1e-4;
            sum(bwRatio) <= 1;
            tEdge >= 0;
            for g = 1:p.G
                users = find(beta(g, :) > 0);
                for u = users
                    tEdge(g) >= minTimeAtFullBandwidth(g, u) * inv_pos(bwRatio(g));
                    latencyCcs(g) + lUpdEdge(g) + ...
                        tEdge(g) + lUpdUav(u) <= p.tau;
                end
            end
    cvx_end
    require_cvx_success(cvx_status, 'bandwidth allocation');
end

function pEdge = solve_edge_power_cvx(tPowerTarget, bwRatio, beta, hEdgeUav, p)
    gainOverNoise = abs(hEdgeUav).^2 / p.noisePower;
    requiredPower = zeros(p.G, p.U);
    for g = 1:p.G
        for u = 1:p.U
            if beta(g, u) > 0
                reqSnr = 2^(p.dataEdgeBits / (bwRatio(g) * p.B * tPowerTarget(g))) - 1;
                requiredPower(g, u) = reqSnr / gainOverNoise(g, u);
            end
        end
    end

    cvx_begin quiet
        variable pEdge(p.G)
        minimize(tPowerTarget(:)' * pEdge)
        subject to
            0 <= pEdge <= p.pEdgeMax;
            for g = 1:p.G
                users = find(beta(g, :) > 0);
                for u = users
                    pEdge(g) >= requiredPower(g, u);
                end
            end
    cvx_end
    require_cvx_success(cvx_status, 'edge transmit power');
end

function tBudget = edge_time_budgets(latencyCcs, lUpdEdge, lUpdUav, beta, p)
    tBudget = inf(p.G, 1);
    for g = 1:p.G
        users = find(beta(g, :) > 0);
        for u = users
            candidate = p.tau - latencyCcs(g) - lUpdEdge(g) - lUpdUav(u);
            tBudget(g) = min(tBudget(g), candidate);
        end
        if ~isfinite(tBudget(g)) || tBudget(g) <= 0
            error('No positive edge transmission time budget for edge %d.', g);
        end
    end
end

function tEdge = edge_transmission_times(beta, bwRatio, pEdge, hEdgeUav, p)
    snr = link_snr(pEdge, hEdgeUav, p);
    rates = bwRatio(:) .* p.B .* log2(1 + snr);
    tEdge = zeros(p.G, 1);
    for g = 1:p.G
        users = find(beta(g, :) > 0);
        if isempty(users)
            tEdge(g) = 0;
        else
            tEdge(g) = max(p.dataEdgeBits ./ rates(g, users));
        end
    end
end

function snr = link_snr(pEdge, hEdgeUav, p)
    snr = (pEdge(:) .* abs(hEdgeUav).^2) / p.noisePower;
end

function [objective, maxTotalLatency] = evaluate_objective( ...
    powerCcs, latencyCcs, pEdge, tEdge, alphaEdge, alphaUav, ...
    lUpdEdge, lUpdUav, beta, p)
    ccsTransmitEnergy = powerCcs * max(latencyCcs);
    edgeComputeEnergy = sum(alphaEdge(:) .* p.pGpuEdge .* lUpdEdge(:));
    edgeTransmitEnergy = sum(pEdge(:) .* tEdge(:));
    uavComputeEnergy = sum(alphaUav(:) .* p.pGpuUav .* lUpdUav(:));
    objective = ccsTransmitEnergy + edgeComputeEnergy + ...
        edgeTransmitEnergy + uavComputeEnergy;

    totalLatency = zeros(p.G, p.U);
    for g = 1:p.G
        for u = find(beta(g, :) > 0)
            totalLatency(g, u) = latencyCcs(g) + ...
                lUpdEdge(g) + tEdge(g) + lUpdUav(u);
        end
    end
    maxTotalLatency = max(totalLatency(:));
end

function plot_convergence(objHistory, outDir)
    iterations = 1:numel(objHistory);
    figure('Color', 'w');
    plot(iterations, objHistory, '-o', ...
        'LineWidth', 1.8, 'MarkerSize', 5, 'MarkerFaceColor', [0.10, 0.45, 0.85]);
    grid on;
    xlabel('BCD iteration');
    ylabel('Objective value (J)');
    title('Objective convergence for LoRA deployment');
    set(gca, 'FontName', 'Times New Roman', 'FontSize', 12);
    exportgraphics(gcf, fullfile(outDir, 'lora_objective_convergence.png'), 'Resolution', 300);
    savefig(gcf, fullfile(outDir, 'lora_objective_convergence.fig'));
end

function require_cvx_success(status, blockName)
    ok = strcmpi(status, 'Solved') || strcmpi(status, 'Inaccurate/Solved');
    if ~ok
        error('%s CVX block failed with status: %s', blockName, status);
    end
end
