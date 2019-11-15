%{

Render samples_per_batch many random images per batch
Requires the variable batch to be set ahead of the time. batch > 0.

Example call using bash script.

#!/bin/bash
matlab  -singleCompThread -r "batch=1; warning('off', 'all'); batch_render"

%}

if exist('batch', 'var') == 0
    disp('You have to set a batch number by setting the variable batch. batch > 0');
    return;
end;

% for_identity_db is a switch: 0 indicates regression to latents (to train EIG), 1 indicates for identity classification (to train or fine-tune VGG).
for_identity_db = 0;
samples_per_batch = 400;

ROOT = getenv('PYTHONPATH'); % ROOT of the project folder
ROOT = split(ROOT, ':');
ROOT = ROOT{end};

baselpath = strcat(ROOT, '/bfm09-generator', '/bfm_utils/PublicMM1');
baselmatlabpath = strcat(baselpath, '/matlab');
addpath(baselpath);
addpath(baselmatlabpath);

OUTDIR = strcat('./output/batch-render/')
mkdir(OUTDIR);
PARAMSDIR = strcat(OUTDIR, 'coeffs/');
mkdir(PARAMSDIR);

coeffs = load('training_db_latents.mat');
coeffs = coeffs.coeffs;
rng shuffle;

[model, msz] = load_model();
DIM = 50; % length of dimension per part -- total dimensionality 50*4=200 for shape and 50*4=200 for texture.
% set general rendering parameters
rp = defrp;
rp.dir_light.dir = [0;1;1];
rp.dir_light.intens = .6.*ones(3,1);
rp.width = 227;
rp.height=227;

batch_begin_index = (batch - 1) * samples_per_batch; 
batch_end_index = batch * samples_per_batch;
offset = 0;
disp(batch);

for i = batch_begin_index:batch_end_index-1
    disp(i);
    try
 	imread(strcat(OUTDIR, int2str(i), '.png'));
        continue;
    catch
  	%obtain rendering parameters
        viewing_params = zeros(1,4);
        viewing_params(1) = rand()*3-1.5; 
        viewing_params(2) = rand()*1.5-.75;
        viewing_params(3) = rand()*2-1.; % multiply by 80
        viewing_params(4) = rand()*2-1; % multiply by 80
 
	if for_identity_db == 1 
            shape = coeffs(batch, 1:200);
            texture = coeffs(batch, 201:400);
	else
            shape = randn(1, 200);
            texture = randn(1, 200);
	end;

        all_params = [shape, texture, viewing_params];

	alpha = reshape(shape, [DIM,4]);
        beta = reshape(texture, [DIM,4]);

        rp.phi = viewing_params(1);
        rp.elevation = viewing_params(2);
        rp.mode_az = viewing_params(3)*80;
        rp.mode_el = viewing_params(4)*80;
        rp.width = 227;rp.height=227;

        [shape, tex, tl] = coeffs_to_head(alpha, beta, DIM, 1);
	handle = display_face(shape, tex, tl, rp, rp.mode_az, rp.mode_el, 6);
        img = hardcopy(handle, '-dopengl', '-r300');
        img = imresize(img, [227, 227]);
        imwrite(img, strcat(OUTDIR, int2str(i), '.png'));
        dlmwrite(strcat(PARAMSDIR, int2str(i), '.txt'), all_params);

      end;
end;
