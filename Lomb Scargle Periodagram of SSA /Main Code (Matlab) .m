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



% Step 3: Save the selected columns to a new Excel file (optional)
% This part can be performed in MATLAB using writetable or xlswrite
writetable(monthly_data, 'file1.xlsx');  % Example saving the table to an Excel file
