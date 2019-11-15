ROOT = getenv('PYTHONPATH'); % ROOT of the project folder
ROOT = split(ROOT, ':');
ROOT = ROOT{end};

baselpath = strcat(ROOT, '/bfm09-generator', '/bfm_utils/PublicMM1');
baselmatlabpath = strcat(baselpath, '/matlab');
addpath(baselpath);
addpath(baselmatlabpath);

load('../neural_analysis/mask_faces.mat');
mask = tr;

rng shuffle;

[model, msz] = load_model();

OUTDIR = './output/renderings/out_of_sample/'
filename = './output/infer_output.hdf5';

network_features = h5read(filename, '/latents');
network_features = transpose(network_features);
network_features = network_features/10;
filenames = h5read(filename, '/filenames');
attendedfaces = h5read(filename, '/Att');
attendedfaces = transpose(attendedfaces);

N = size(network_features);
N = N(1);

mkdir(OUTDIR);
DIM = 50;

rp     = defrp;
rp.dir_light.dir = [0;1;1];
rp.dir_light.intens = .6.*ones(3,1);
rp.mode_az = 0;
rp.mode_el = 0;
rp.phi = 0;
rp.elevation = 0;
rp.width = 300; rp.height=300;

% mode_az, mode_el, phi, elevation
set_views = [[0, 0, 0, 0]; [50, 0, 0, 0]; [-50, 0, 0, 0]; [0, 0, 1.5, 0]; [0, 0, 0.75, 0]; [0, 0, -1.5, 0]; [0, 0, -0.75, 0]];

for i = 1:N

    input = network_features(i, :);

    shape = input(1:200);
    texture = input(201:400);
    alpha = reshape(shape, [DIM,4]);
    beta = reshape(texture, [DIM,4]);
    [shape, tex, tl] = coeffs_to_head(alpha, beta, DIM, 1);

    for k = 1:7
        rp.mode_az = set_views(k, 1);
        rp.mode_el = set_views(k, 2);
        rp.phi = set_views(k, 3);
        rp.elevation = set_views(k, 4);


        handle = display_face(shape, tex, model.tl, rp, rp.mode_az, rp.mode_el, []);
        img = hardcopy(handle, '-dopengl', '-r300');
        img = imresize(img, [300, 300]);
        imwrite(img, strcat(OUTDIR, int2str(i), '_', int2str(k), '.png'), 'png');

    end;

    rp.mode_az = input(403)*80;
    rp.mode_el = input(404)*80;
    rp.phi = input(401); %0.0;
    rp.elevation = input(402); 

    handle = display_face(shape, tex, model.tl, rp, rp.mode_az, rp.mode_el, []);
    img = hardcopy(handle, '-dopengl', '-r300');
    img = imresize(img, [300, 300]);
    imwrite(img, strcat(OUTDIR, int2str(i), '_render.png'), 'png');

    attend = attendedfaces(i, :);
    attend = reshape(attend, 227, 227, 3);
    attend = uint8(attend);
    attend = permute(attend, [2, 1, 3]);
    imwrite(attend, strcat(OUTDIR, int2str(i), '_attended.png'), 'png');

end;
