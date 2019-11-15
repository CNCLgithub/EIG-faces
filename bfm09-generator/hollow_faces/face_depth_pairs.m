%{
-- Generate a large pool of face-depth pairs: (1) illusory images by suppressing the depth and (2) control by matching the light source location to its perceived position for that level of depth-suppression (based on the lighting direction judgment task results).
-- Afterward we choose 6 identities for each of the 9 levels that reflect the depth statistics -- mean and variance -- for each level.
%}

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
rp.mode_az = 0.0;
rp.phi = 0.0;
rp.elevation = 0.0;

OUTDIR = './output/face-depth-pairs/';
mkdir(OUTDIR);

coeffs = load('random_latents.mat');
random_coeffs = coeffs.random_coeffs;

load('nose_mouth.mat');
load('neck_ears.mat');
orig_tl = model.tl;
% remove neck and ears
orig_tl = orig_tl(tr == 1, :);
% remove inside of nose and mouth
tr(faces == 0) = 0;
tl = model.tl(tr == 1, :);

flatten = [1, 0.75, 0.50, 0.25, 0, -0.25, -0.50, -0.75, -1];
illusory_matching =  [75.0, 56.25, 37.5, 18.75, 0, -56.25, -56.25, -56.25, -56.25];

counter = 1;

for i = 1:90

  for j = 1:9
    disp(counter);
    latents = random_coeffs(counter,:);
    alpha = latents(1:200);
    beta = latents(201:400);

    alpha = reshape(alpha, [DIM, 4]);
    beta = reshape(beta, [DIM, 4]);
    [shape, tex, back_tl] = coeffs_to_head(alpha, beta, DIM, 1, model, msz);
    M = length(shape);

    flattened = shape;
    flattened(3:3:M) = flatten(j)*shape(3:3:M);

    % the difference image
    rp.mode_el = 75.0;
    handle = display_face(shape, tex, orig_tl, rp, rp.mode_az, rp.mode_el, 6);
    source_img = hardcopy(handle, '-dopengl', '-r300');
    source_img = imresize(source_img, [227, 227]);
    handle = display_face(shape, tex, tl, rp, rp.mode_az, rp.mode_el, 6);
    dest_img = hardcopy(handle, '-dopengl', '-r300');
    dest_img = imresize(dest_img, [227, 227]);
    diff_img = double(source_img) - double(dest_img);

    % regular shape with illusory matching light
    rp.mode_el = illusory_matching(j);
    handle = display_face(shape, tex, orig_tl, rp, rp.mode_az, rp.mode_el, 6);
    regular_img = hardcopy(handle, '-dopengl', '-r300');
    regular_img = imresize(regular_img, [227, 227]);
    imwrite(regular_img, strcat(OUTDIR, int2str(counter), '_', int2str(j), '_regular.png'), 'png');

    % with compressed shape
    rp.mode_el = 75.0;
    handle = display_face(flattened, tex, tl, rp, rp.mode_az, rp.mode_el, 6);
    img = hardcopy(handle, '-dopengl', '-r300');
    img = imresize(img, [227, 227]);
    img = double(img) + diff_img;
    mask = 1 - (sum(regular_img, 3) == 765);
    mask = repmat(mask, [1, 1, 3]);
    img = img .* mask + (1 - mask) * 255;


    % clean up any artifact
    if j > 1
       temp = sum(img(71:140, 71:140,:), 3);
       for k = 1:20
	   [a, b] = max(temp(:));
	   [x, y] = ind2sub(size(temp), b);
	   x = x + 70; y = y + 70;
	   img(x, y, :) = (img(x-2, y, :) + img(x+2, y, :) + img(x, y-2, :) + img(x, y+2, :))/4.;
	   temp(x - 70, y - 70) = sum(img(x,y,:));
       end;
    end;

    imwrite(uint8(img), strcat(OUTDIR, int2str(counter), '_', int2str(j), '_relief.png'), 'png');
    
    counter = counter + 1;
  end;

end;


