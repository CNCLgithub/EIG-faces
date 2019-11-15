% Generate a random identity (shape and texture properties) and render it at 20 random scene configurations (ligthing and viewpoint).

ROOT = getenv('PYTHONPATH'); % ROOT of the project folder
ROOT = split(ROOT, ':');
ROOT = ROOT{end};

baselpath = strcat(ROOT, '/bfm09-generator', '/bfm_utils/PublicMM1');
baselmatlabpath = strcat(baselpath, '/matlab');
addpath(baselpath);
addpath(baselmatlabpath);

rng shuffle;
[model, msz] = load_model();
DIM = 50; % length of dimension per part -- total dimensionality 50*4=200 for shape and 50*4=200 for texture.
% set general rendering parameters
rp = defrp;
rp.dir_light.dir = [0;1;1];
rp.dir_light.intens = .6.*ones(3,1);
rp.width = 227;
rp.height=227;

OUTDIR = './output/random_samples/';
mkdir(OUTDIR);
PARAMSDIR = strcat(OUTDIR, 'coeffs/');
mkdir(PARAMSDIR);

% get a random identity
shape = randn(1, 200);
texture = randn(1, 200);

% generate 20 amples each with a randomy drawn scene configuration (lighting and viewpoint)
for i=1:20
    disp(i);
    try
        imread(strcat(OUTDIR, int2str(i-1), '.png'));
        continue;
    catch
        %obtain rendering parameters
        viewing_params = zeros(1,4);
        viewing_params(1) = rand()*3-1.5; 
        viewing_params(2) = rand()*1.5-.75;
        viewing_params(3) = rand()*2-1.; % multiply by 80
        viewing_params(4) = rand()*2-1; % multiply by 80

        all_params = [shape, texture, viewing_params];
   
        rp.phi = viewing_params(1);
        rp.elevation = viewing_params(2);
        rp.mode_az = viewing_params(3)*80;
        rp.mode_el = viewing_params(4)*80;

        alpha = reshape(shape, [DIM,4]);
        beta = reshape(texture, [DIM,4]);

        [shape_, tex_, tl] = coeffs_to_head(alpha, beta, DIM, 1);
        handle = display_face(shape_, tex_, tl, rp, rp.mode_az, rp.mode_el, 6);

        img = hardcopy(handle, '-dopengl', '-r300');
        img = imresize(img, [227, 227]);

        imwrite(img, strcat(OUTDIR, int2str(i-1), '.png'));
        dlmwrite(strcat(PARAMSDIR, int2str(i-1), '.csv'), all_params);
     end;
end;
