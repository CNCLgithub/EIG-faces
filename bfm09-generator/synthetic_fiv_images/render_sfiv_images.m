% Generate synthetic FIV imageset. THis is 175 images mimicking the FIV imageset in terms of the number of identiites (25) and their viewpoints (7 poses from left-profile to right-profile).
% set the batch variable for new synthetic FIV imagesets.

batch = 1;

ROOT = getenv('PYTHONPATH'); % ROOT of the project folder
ROOT = split(ROOT, ':');
ROOT = ROOT{end};

baselpath = strcat(ROOT, '/bfm09-generator', '/bfm_utils/PublicMM1');
baselmatlabpath = strcat(baselpath, '/matlab');
addpath(baselpath);
addpath(baselmatlabpath);

OUTDIR = ['./output/sfiv-imagesets/', num2str(batch), '/'];
mkdir(OUTDIR);

rng shuffle;
[model, msz] = load_model();
DIM = 50; % length of dimension per part -- total dimensionality 50*4=200 for shape and 50*4=200 for texture.
% set general rendering parameters
rp = defrp;
rp.dir_light.dir = [0;1;1];
rp.dir_light.intens = .6.*ones(3,1);
rp.mode_az = 0;
rp.mode_el = 0;
rp.width = 224;
rp.height=224;

disp(batch);

phis = [0, 0.75, 1.5, -0.75, -1.5, 0, 0];
elvs = [0, 0, 0, 0, 0, -.5, .5];

coeffs = dlmread('sfiv_latents.txt');

counter = 1;
for i=1:25
    input = coeffs(i, :);
    shape = input(1:200);
    texture = input(201:400);
    alpha = reshape(shape, [DIM,4]);
    beta = reshape(texture, [DIM,4]);
    noisy_pose = randn(1,7)*0.05;
    noisy_pose(3) = abs(noisy_pose(3))*-1;noisy_pose(5) = abs(noisy_pose(5));
    noisy_phis = phis + noisy_pose;
    noisy_pose = randn(1,7)*0.05;
    noisy_elvs = elvs + noisy_pose;

    for j = 1:7
        rp.phi = noisy_phis(j);
        rp.elevation = noisy_elvs(j);

        [shape, tex, tl] = coeffs_to_head(alpha, beta, DIM, 1);
	handle = display_face(shape, tex, tl, rp, rp.mode_az, rp.mode_el, []);

        img = hardcopy(handle, '-dopengl', '-r300');
        img = imresize(img, [224, 224]);
        imwrite(img, strcat(OUTDIR, int2str(counter), '.png'));
        disp(counter);
        counter = counter + 1;
    end;
end;
