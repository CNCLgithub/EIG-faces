ROOT = getenv('PYTHONPATH'); % ROOT of the project folder
ROOT = split(ROOT, ':');
ROOT = ROOT{end};

baselpath = strcat(ROOT, '/bfm09-generator', '/bfm_utils/PublicMM1');
baselmatlabpath = strcat(baselpath, '/matlab');
addpath(baselpath);
addpath(baselmatlabpath);

rng shuffle;

bfm = 0; % 0 for FIV images; 1 for BFM images

load('./mask_faces.mat');
mask = tr;

[model, msz] = load_model();

if bfm == 1
    OUTDIR = './output/intrinsics/bfm/';
    filename = './output/eig_classifier_bfm.hdf5';
else
    OUTDIR = './output/intrinsics/fiv/';
    filename = './output/eig_classifier_fiv.hdf5';
end;

network_features = h5read(filename, '/2');
network_features = transpose(network_features);
network_features = network_features/10;

mkdir(OUTDIR);
DIM = 50;

phis = [0, 0.75, 1.5, -0.75, -1.5, 0, 0];
elvs = [0, 0, 0, 0, 0, -.5, .5];

rp     = defrp;
rp.dir_light.dir = [0;1;1];
rp.dir_light.intens = .6.*ones(3,1);
rp.mode_az = 0;
rp.mode_el = 0;
rp.width = 227; rp.height=227;

counter = 1;
for i=1:25
    input = network_features(i, :);

    shape = input(1:200);
    texture = input(201:400);
    alpha = reshape(shape, [DIM,4]);
    beta = reshape(texture, [DIM,4]);
    [shape, tex, tl] = coeffs_to_head(alpha, beta, DIM, 1);

    for j = 1:7
        rp.phi = phis(j);
        rp.elevation = elvs(j);

	% albedo
	handle = display_intrinsics(shape, tex, tl, rp, rp.mode_az, rp.mode_el, 2, mask, 1);
        img = hardcopy(handle, '-dopengl', '-r300');
        img = imresize(img, [227, 227]);
        imwrite(img, strcat(OUTDIR, int2str(counter), '_albedo.png'));

	% normals
	handle = display_intrinsics(shape, tex, tl, rp, rp.mode_az, rp.mode_el, 3, mask, 1);
        img = hardcopy(handle, '-dopengl', '-r300');
        img = imresize(img, [227, 227]);
        imwrite(img, strcat(OUTDIR, int2str(counter), '_normals.png'));

        disp(counter);
        counter = counter + 1;
    end;
end;

