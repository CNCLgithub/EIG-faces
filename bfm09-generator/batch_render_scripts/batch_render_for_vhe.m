%{

Render samples_per_batch images of a new random identity at random scene configurations (default 20 images) per batch.
Requires the variable batch to be set ahead of the time. batch > 0.

Example call using bash script.

#!/bin/bash
matlab  -singleCompThread -r "batch=1; warning('off', 'all'); batch_render_for_vhe"

%}

if exist('batch', 'var') == 0
    disp('You have to set a batch number by setting the variable batch. batch > 0');
    return;
end;

samples_per_identity = 20;

ROOT = getenv('PYTHONPATH'); % ROOT of the project folder
ROOT = split(ROOT, ':');
ROOT = ROOT{end};

baselpath = strcat(ROOT, '/bfm09-generator', '/bfm_utils/PublicMM1');
baselmatlabpath = strcat(baselpath, '/matlab');
addpath(baselpath);
addpath(baselmatlabpath);

OUTDIR = strcat('./output/batch-render-vhe/');
mkdir(OUTDIR);
MYFOLDER = strcat(OUTDIR, int2str(batch), '/')
mkdir(MYFOLDER);
rng shuffle;

[model, msz] = load_model();

DIM = 50;
disp(batch);

% setup the general rendering environment
rp = defrp;
rp.dir_light.dir = [0;1;1];
rp.dir_light.intens = .6.*ones(3,1);
rp.width = 227;
rp.height=227;


% get identity
shape = randn(1, 200);
texture = randn(1, 200);

params = [shape, texture];

for i=1:samples_per_identity
    disp(i);
    try
        imread(strcat(MYFOLDER, int2str(i-1), '.png'));
        continue;
    catch
        %obtain rendering parameters
        viewing_params = zeros(1,4);
        viewing_params(1) = rand()*3-1.5;
        viewing_params(2) = rand()*1.5-.75;
        viewing_params(3) = rand()*2-1.; % multiply by 80
        viewing_params(4) = rand()*2-1; % multiply by 80

        all_params = [shape, texture, viewing_params];

        alpha = reshape(shape, [DIM,4]);
        beta = reshape(texture, [DIM,4]);

        rp.phi = viewing_params(1);
        rp.elevation = viewing_params(2);
        rp.mode_az = viewing_params(3)*80;
        rp.mode_el = viewing_params(4)*80;

        [shape_, tex_, tl] = coeffs_to_head(alpha, beta, DIM, 1);
        handle = display_face(shape_, tex_, tl, rp, rp.mode_az, rp.mode_el, 6);

        img = hardcopy(handle, '-dopengl', '-r300');
        img = imresize(img, [227, 227]);

        imwrite(img, strcat(MYFOLDER, int2str(i-1), '.png'));
        dlmwrite(strcat(MYFOLDER, int2str(i-1), '.csv'), all_params);
    end;
end;
