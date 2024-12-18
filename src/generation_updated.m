clc;
clear;

%%% Simulate ARA experiment

%%%%%%%%%%%%%%%%%%%%%%% Stage 1: Channel model construction %%%%%%%%%%%%%%%%
% Set RNG for repeatability
s = rng(42);

% Antenna and Layout Parameters
N = 14;                 % Antennas per sector (7 radios x 2 polarizations)
d = 0.27;               % Spacing (meters)
Nu = 2;                 % Antennas per UEs (2 links which is defined as channels in the ARA testbed)
du = 0.14;              % User antenna spacing (meters)
f_c = 563e6;            % Main Carrier frequency (Hz)
range = 1550;           % Maximum range of the area (meters)
UserDistance = 400      % in meters

% Antenna Array Definitions
AA(1) = winner2.AntennaArray('ULA', 3*N, d);  % Base station antennas (42 elements total)
AA(2) = winner2.AntennaArray('ULA', Nu, du);  % Mobile station antennas

% Layout Configuration
BS_indexSingleSector = {1};          % Single macrobase station
MSindexOne = [2];                    % Single Mobile station
numLinks = 1;                        % Number of links

configLayoutSingleLink = winner2.layoutparset(MSindexOne, BS_indexSingleSector, numLinks, AA, range);

% Explicitly set the number of antennas for each station
configLayoutSingleLink.Stations(1).NofAntennas = 3*N;  % Base station (42 antennas)
configLayoutSingleLink.Stations(2).NofAntennas = Nu;   % Mobile station (2 antennas)

% Pairing and Propagation Scenario
configLayoutSingleLink.Pairing = [1; 2];                 % Pair the mobile station to the base station
configLayoutSingleLink.ScenarioVector = [14];            % Scenario 14: Macrocell Rural (D1)
configLayoutSingleLink.PropagConditionVector = [0];      % LOS (Line of Sight)

% Translate positions to ensure nonnegative values
offset_x = 7500537.5;  % Offset to make positions nonnegative
offset_y = -4670000;   % Offset to make positions nonnegative

% Node Positions
configLayoutSingleLink.Stations(1).Pos(1:2) = [0.0; 0.0]; % Base station position
configLayoutSingleLink.Stations(2).Pos(1:2) = [UserDistance; UserDistance]; % Mobile station position

% Calculate distance and azimuth early
bs_pos = configLayoutSingleLink.Stations(1).Pos(1:2);
ue_pos = configLayoutSingleLink.Stations(2).Pos(1:2);
distance_vector = ue_pos - bs_pos;
distance_to_base_station = norm(distance_vector);
azimuth = atan2d(distance_vector(2), distance_vector(1));

% WINNER II Model Parameters
frameLen = 2080;                  % Number of time samples
cfgWim = winner2.wimparset;       % Default parameters

% Customize WINNER II Parameters
cfgWim.NumTimeSamples = frameLen;
cfgWim.FixedPdpUsed = "no";
cfgWim.FixedAnglesUsed = "no";
cfgWim.IntraClusterDsUsed = "yes";
cfgWim.DelaySpread = 0.5e-6;
cfgWim.CenterFrequency = f_c;
cfgWim.UniformTimeSampling = "yes";
cfgWim.ShadowingModelUsed = "yes";
cfgWim.PathLossModelUsed = "yes";
cfgWim.RandomSeed = 31415926;     % Ensure repeatability

% Create the WINNER II Channel System Object
WinnerChannel_Sim_ARA_Experiment = comm.WINNER2Channel(cfgWim, configLayoutSingleLink);

%%%%%%%%%%%%%%% End of Stage 1: Channel model construction %%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%  Stage 2: CSI generation %%%%%%%%%%%%%%%%%%%%%    
% Display Channel Information
channelInfo = info(WinnerChannel_Sim_ARA_Experiment);
disp(channelInfo);

% Debug output for antenna configuration
disp('Base Station number of antennas:');
disp(configLayoutSingleLink.Stations(1).NofAntennas);
disp('Mobile Station number of antennas:');
disp(configLayoutSingleLink.Stations(2).NofAntennas);

% Create CSI matrix using the actual dimensions
CSI = zeros(frameLen, Nu, 3*N);

% Channel Generation
for bs_ant = 1:3*N
    txSig{1} = (randn(cfgWim.NumTimeSamples, 3*N) + 1j*randn(cfgWim.NumTimeSamples, 3*N))/sqrt(2);
    tx_power = db2pow(43);  % Typical UE transmit power in linear scale
    txSig{1}(:,bs_ant) = sqrt(tx_power) * txSig{1}(:,bs_ant);
    
    % Get channel response
    rx_temp = WinnerChannel_Sim_ARA_Experiment(txSig);
    
    % % Add thermal noise
    % noise_power = db2pow(-174 + 10*log10(20e6) - 30); % -174 dBm/Hz + bandwidth in Hz
    % noise = sqrt(noise_power/2) * (randn(size(rx_temp{1})) + 1j*randn(size(rx_temp{1})));
    % rx_temp{1} = rx_temp{1} + noise;
    
    % Store the response with realistic path loss
    for ms_ant = 1:Nu
        % path_loss = 20*log10(4*pi*distance_to_base_station*f_c/3e8); % Free space path loss
        CSI(:,ms_ant,bs_ant) = rx_temp{1}(:,ms_ant);% * db2pow(-path_loss/2);
    end
end

disp('CSI matrix dimensions:');
disp(size(CSI));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% End of Stage 2: Simulated CSI generation %%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%% Stage 3: Postprocessing of the simulated CSI data to be prepared for SCB training %%%%%%%%%%%%%%%% 
% Prepare static values for CSV columns
numerology_pattern = 0; % The numerology pattern affect the value of carrier frequency

antenna_geometry = "ULA";
base_station_antennas = 3*N;
user_equipment_antennas = Nu;
carrier_frequency = f_c;
antenna_spacing = d;
scenario = "D1";

% Initialize cell array for data with correct size
totalRows = frameLen * Nu * 3*N;
csiData = cell(totalRows, 14); % 14 columns total

% Fill the data array
rowIndex = 1;
for t = 1:frameLen
    for tx = 1:Nu
        for rx = 1:(3*N)
            csiData(rowIndex,:) = {numerology_pattern, ...
                                  antenna_geometry, ...
                                  base_station_antennas, ...
                                  user_equipment_antennas, ...
                                  carrier_frequency, ...
                                  antenna_spacing, ...
                                  scenario, ...
                                  azimuth, ...
                                  distance_to_base_station, ...
                                  t, ...
                                  tx, ...
                                  rx, ...
                                  real(CSI(t, tx, rx)), ...
                                  imag(CSI(t, tx, rx))};
            rowIndex = rowIndex + 1;
        end
    end
end

% Create table with headers
headers = {'numerology_pattern', 'antenna_geometry', 'base_station_antennas', ...
           'user_equipment_antennas', 'carrier_frequency', 'antenna_spacing', ...
           'scenario', 'azimuth', 'distance_to_base_station', 'timeIndex', ...
           'txIndex', 'rxIndex', 'real', 'imag'};
csiTable = cell2table(csiData, 'VariableNames', headers);

% Save CSI data to CSV
csvFileName = 'Simulated_UE_UPlink_CSI_Data_New.csv';
writetable(csiTable, csvFileName);

disp(['UE Uplink CSI data saved to ', csvFileName]);