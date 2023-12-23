function [rmssd, sdnn] = HRV(rri)

% Calculate the root mean square of successive differences (RMSSD).
differences = diff(rri);
rmssd = sqrt(mean(differences.^2));

% Calculate the standard deviation of normal RR intervals (SDNN).
% rr_intervals = rr_intervals[~isnan(rr_intervals)];
sdnn = std(rri);


end
