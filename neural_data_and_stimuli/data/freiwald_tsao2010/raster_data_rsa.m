toolbox_basedir_name = '/Users/ilker/work/ndt.1.0.4/';%'/Users/ilker/work/ndt.1.0.2/';
addpath(toolbox_basedir_name);
add_ndt_paths_and_init_rand_generator

%patch = 'MLMF';
%patch = 'AL';
patch = 'AM';

rd_dir = ['/Users/ilker/work/DATASETS/Doris_Winrich_2010_science_data/FV data/raster_data_fv_' patch]; %MLMF AL AM

save_prefix_name = '/Users/ilker/work/DATASETS/Doris_Winrich_2010_science_data/FV data/binned/binned';
bin_width = 200;
step_size = 50;
start_time = 150;
end_time = 400;
create_binned_data_from_raster_data(rd_dir, save_prefix_name, bin_width, step_size, start_time, end_time);

binned_data_filename = strcat(save_prefix_name, '_',int2str(bin_width), 'ms_bins_', int2str(step_size), ...
    'ms_sampled_', int2str(start_time), 'start_time_', int2str(end_time), 'end_time.mat');
%'/Users/ilker/work/DATASETS/Doris_Winrich_2010_science_data/FV data/binned/binned_100ms_bins_50ms_sampled_1start_time_800end_time.mat';
binned_data = load(binned_data_filename);

[stimulus_time_average_activity_matrix site_time_stimulus_activity_matrix] = get_average_population_activity_from_binned_data(binned_data_filename, 'stimID', 2);

V = 1;
neural_vecs = squeeze(site_time_stimulus_activity_matrix(:, V, :));
neural_vecs = neural_vecs';

%suff_repeat = [];
%for k = 1:200
%    inds_of_sites_with_at_least_k_repeats = find_sites_with_k_label_repetitions(binned_data.binned_labels.stimID, k);
%    suff_repeat = [suff_repeat inds_of_sites_with_at_least_k_repeats];
%    num_sites_with_k_repeats(k) = length(inds_of_sites_with_at_least_k_repeats);
%end

%atleastonce = [];
%for k = 1:214
%    if sum(suff_repeat == k) == 0
%        continue;
%    else
%        atleastonce = [atleastonce k];
%    end
%end

inds_of_sites_with_at_least_k_repeats = find_sites_with_k_label_repetitions(binned_data.binned_labels.stimID, 1);
disp(size(inds_of_sites_with_at_least_k_repeats));
disp(size(neural_vecs));
neural_vecs = neural_vecs(:, inds_of_sites_with_at_least_k_repeats);

save(['neural_vecs_' patch '.mat'], 'neural_vecs');



sim = zeros(175,175);
for i = 1:175
    for j = 1:175
        tmp = corrcoef(neural_vecs(i,:), neural_vecs(j,:));
        sim(i,j) = tmp(1,2);
    end;
    sim(i,i) = 0.2;
end;
sim = sim([51:75 26:50 1:25 76:100 101:175],:);
sim = sim(:, [51:75 26:50 1:25 76:100 101:175]);



%dlmwrite(strcat(int2str(step_size*V),'_', int2str(step_size), '_', patch, '_v2.txt'), neural_corrs, 'delimiter', ' ');
figure;
imagesc(sim(175:-1:1, :));
colormap gray;
cmap = colormap;
cmap = flipud(cmap);
colormap(cmap);




