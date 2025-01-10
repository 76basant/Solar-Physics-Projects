% objective of this code:
%to plot Lomb Scargle periodagram of SSA from cycles 21 to 24

% Define the input file path
file_path = 'b10.dat';

% Read data from file as a table
data = readtable(file_path, 'FileType', 'text', 'Delimiter', ' ', 'MultipleDelimsAsOne', true);

% Rename columns
data.Properties.VariableNames = {'Year', 'Month', 'SSA'};

% Select data from cycles 21 to 24 (rows 1222 to 1747)
data = data(1161:1686, :);

% Display data
disp(data);

% Calculate monthly averages (if needed)
% For this example, it's assumed the data is already monthly
monthly_data = data;

% Find rows with NaN values
nan_rows = monthly_data(any(ismissing(monthly_data), 2), :);

% Display rows with NaN values
disp('Rows with NaN values:');
disp(nan_rows);

% Add a 'time' column
monthly_data.time = monthly_data.Year + monthly_data.Month / 12;

% Plotting Raw Data of SSA from Cycles 21 to 24
scatter(monthly_data.time, monthly_data.SSA, 5, 'b');
hold on;

% Retrieve current y-ticks
yticks = get(gca, 'YTick');
y_step = yticks(2) - yticks(1);
disp('Y-Ticks:');
disp(yticks);
disp('Y-Step:');
disp(y_step);

% Retrieve current x-ticks
xticks = get(gca, 'XTick');
x_step = xticks(2) - xticks(1);
disp('X-Ticks:');
disp(xticks);
disp('X-Step:');
disp(x_step);

% Data: Year and corresponding cycle numbers
years = [2008 + 12 / 12.0, 1996 + 8 / 12.0, 1986 + 9 / 12.0, 1976 + 3 / 12.0];
cycles = 24:-1:21;

% Plotting vertical lines with annotations
for i = 1:length(years)
    x = years(i);
    cycle = cycles(i);
    line([x x], [0 max(monthly_data.SSA) + y_step / 2], 'Color', 'r', 'LineStyle', '--', 'LineWidth', 0.5);
    text(x + 1, max(monthly_data.SSA) + y_step / 5, sprintf('Cycle %d', cycle), ...
         'Color', 'r', 'FontSize', 12, 'VerticalAlignment', 'bottom');
end

% Set plot limits
ylim([0, max(monthly_data.SSA) + y_step / 2]);
xlim([min(monthly_data.time) - x_step / 5, max(monthly_data.time) + x_step / 5]);

% Add labels and title
xlabel('Time');
ylabel('SSA (\mu Hemi)');
title('SSA Monthly Average Cycles 21-24');

% Display the plot
hold off;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Save the selected columns to a new Excel file (optional)
% This part can be performed in MATLAB using writetable or xlswrite

writetable(monthly_data, 'file1.xlsx');  % Example saving the table to an Excel file



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Define the file path
file_path = 'file1.xlsx';

% Load the Excel file
try
    data = readtable(file_path);
    disp('File loaded successfully.');
catch ME
    disp(['Error: ', ME.message]);
    return;
end

% Ensure the 'Year' and 'Month' columns are integers
data.Year = round(data.Year);
data.Month = round(data.Month);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Ask the user to enter a cycle number (21, 22, 23, or 24)
cycle_Number = input('Enter the cycle number (21, 22, 23, or 24): ');

% Check if the entered cycle number is valid
if cycle_Number == 21
    % Select data for cycle 21 (e.g., rows 1-130)
    dataSelected = data(1:127, :);  % (3/1976-9/1986)
    
    % Create a time variable in months since the start date
    start_date = datetime(1976, 3, 1);
    dataSelected.Date = datetime(dataSelected.Year, dataSelected.Month, 1);
    dataSelected.Time = calmonths(between(start_date, dataSelected.Date, 'months'));

    % Extract time and signal for Lomb-Scargle analysis
     
    % Define the cycle number as a variable
     cycleNumber = 21;

    disp('Data for Cycle 21 has been selected.');

elseif cycle_Number == 22
    % Select data for cycle 22 (e.g., rows 131-249)
    dataSelected = data(127:246, :); % (9/1986-8/1996)
    disp('Data for Cycle 22 has been selected.');
    
    % Create a time variable in months since the start date
    start_date = datetime(1986, 9, 1);
    % Define the cycle number as a variable
    cycleNumber = 22;
    
elseif cycle_Number == 23
    % Select data for cycle 23 (e.g., rows 250-397)
    dataSelected = data(246:394, :); % (8/1996-9/2008)
    disp('Data for Cycle 23 has been selected.');
        
    % Create a time variable in months since the start date
    start_date = datetime(1996, 8, 1);
    
elseif cycle_Number == 24
    % Select data for cycle 24 (e.g., rows 398-end)
    dataSelected = data(394:end, :); % (9/2008-12/2019)
    disp('Data for Cycle 24 has been selected.');
    
    % Create a time variable in months since the start date
    start_date = datetime(2008, 9, 1);
   % Define the cycle number as a variable
    cycleNumber = 24;
    
else
    % Invalid input
    disp('Incorrect input! Please enter a valid cycle number (21, 22, 23, or 24).');
    return; % Exit the script if the input is invalid
end

% Display the selected data
disp('Selected Data:');
disp(dataSelected);

dataSelected.Date = datetime(dataSelected.Year, dataSelected.Month, 1);
    dataSelected.Time = calmonths(between(start_date, dataSelected.Date, 'months'));

    % Extract time and signal for Lomb-Scargle analysis
    time = dataSelected.Time;

Column_Selected = dataSelected.SSA
Component_Name = 'SSA'

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
signal = Column_Selected;

% Frequency range for Lomb-Scargle
i = 1 / length(signal); % Maximum of observed periodicities (min frequency)
j = 1 / 2;              % Minimum of observed periodicities (max frequency)
frequency = linspace(i, j, 300);

% Lomb-Scargle periodogram
[power, ~] = plomb(signal, time, frequency);

% Calculate the periods (1/frequency)
periods = 1 ./ frequency;

% Find peaks in the Lomb-Scargle power spectrum
threshold_95 = prctile(power, 95);
threshold_99 = prctile(power, 99);
[peaks, locs] = findpeaks(power, 'MinPeakHeight', threshold_95);
peak_frequencies = frequency(locs);
peak_periods = 1 ./ peak_frequencies;

% Display detected periodicities
disp('Detected Periodicities:');
for k = 1:length(peak_frequencies)
    fprintf('Frequency: %.5f, Period: %.2f years, Power: %.2f\n', ...
        peak_frequencies(k), peak_periods(k) / 12, power(locs(k)));
end

% Plot the Lomb-Scargle periodogram
figure(1)
%figure('Position', [100, 100, 1000, 600]);

% Create a finer lags axis for smooth interpolation
dataLength = length(signal); % Use the signal length as the basis
fineLagsAxis = linspace(min(frequency), max(frequency), dataLength * 10); % 10 times the original points for smoothness
y_smooth = interp1(frequency, power, fineLagsAxis, 'spline');

% Plot smooth curve
plot(fineLagsAxis, y_smooth, '-blue', 'LineWidth', 2); % Smooth curve


% Add confidence level lines
line(xlim, [threshold_95 threshold_95], 'Color', 'r', 'LineStyle', '--', 'LineWidth', 1.2, ...
    'DisplayName', '95% Confidence');
line(xlim, [threshold_99 threshold_99], 'Color', 'k', 'LineStyle', '-.', 'LineWidth', 1.2, ...
    'DisplayName', '99% Confidence');


% Get the y-axis ticks
yticks = get(gca, 'YTick');

% Calculate the step size
ytick_step = yticks(2) - yticks(1);

% Display the step size
disp(['The step size of the y-axis ticks is: ', num2str(ytick_step)]);


% Annotate detected periodicities
for k = 1:length(peak_frequencies)
    %line([peak_frequencies(k), peak_frequencies(k)], [0, max(power)], ...    'Color', 'g', 'LineStyle', ':', 'LineWidth', 1);
    y_position = power(locs(k)); % Assuming 'power' is a vector corresponding to the peaks

    text(peak_frequencies(k), y_position+ ytick_step/1.1, ...
        sprintf('%.2f yr', peak_periods(k) / 12), ...
        'Color', 'r', 'HorizontalAlignment', 'right', ...
        'VerticalAlignment', 'middle', 'Rotation',90, 'FontSize', 16);
end

% Set axis limits
xlim([0.005, 0.189]);
ylim([0, max(power) + ytick_step]);


% Finalize plot
xlabel('Frequency (1/month)');
ylabel('Power');


% Create the title with the component and cycle number as variables
title(['Lomb-Scargle Periodogram of ', Component_Name, ' through cycle ', num2str(cycle_Number)]);

% Update the legend dynamically based on the selected component
legend(Component_Name, '95% Confidence', '99% Confidence', 'Smooth Curve', 'Location', 'northeast');

%grid on;
hold off;


