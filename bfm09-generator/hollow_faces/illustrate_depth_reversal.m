%{
-- Generate depth suppressed images to illustrate the illusion on a continuum of such depth reversals. The resulting images can be put into a video using ffmpeg
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
rp.width = 200;
rp.height=200;
rp.mode_az = 0.0;
rp.mode_el = 75.0; % light above the face and frontally aligned
rp.phi = pi/2;
rp.elevation = 0.0;

OUTDIR = './output/illustrate/';
mkdir(OUTDIR);

a = load('random_latents.mat');
random_coeffs = a.random_coeffs;

flatten = [1, 0.75, 0.50, 0.25, 0, -0.25, -0.50, -0.75, -1];

counter = 1;
step = 50;
flatten = linspace(-1, 1, step);
for i = step:-1:1
    disp(counter);
  
    latents = random_coeffs(4,:) * 0; % mean shape and texture vectors -- we use that norm face to illustrate the illusion here.
    alpha = latents(1:200);
    beta = latents(201:400);

    alpha = reshape(alpha, [DIM, 4]);
    beta = reshape(beta, [DIM, 4]);
    [shape, tex, tl] = coeffs_to_head(alpha, beta, DIM, 1, model, msz);
    M = length(shape);

    flattened = shape;
    flattened(3:3:M) = flatten(i)*shape(3:3:M); % depth suppression

    % with compressed shape
    handle = display_face_to_illustrate_depth_reversal(shape, flattened, tex, tl, rp, rp.mode_az, rp.mode_el, 1);
    img = hardcopy(handle, '-dopengl', '-r300');
    img = imresize(img, [200, 200]);
    img = double(img);
    indices = zeros(80, 2);
    if i > 0
       threshold = 255 * 3;
       offset_x = 107;
       offset_y = 98;
       temp = sum(img(offset_x:140, offset_y:140,:), 3);
       for k = 1:80
           [a, b] = max(temp(:));
           if a < 490 
               threshold = a;
 	       break;
           end;
           [x, y] = ind2sub(size(temp), b);
           indices(k, :) = [x,y];
           temp(x, y) = 0;
        end;

        diff = 2;
        k = k - 1;
        for l = k:-1:1
	    x = indices(l, 1);
 	    y = indices(l, 2);
	    x = x + offset_x - 1; y = y + offset_y - 1;
	    neighbor_sum = img(x, y, :) * 0;
	    m = 0;
 	    if sum(img(x-diff, y, :)) <= threshold
	        neighbor_sum = img(x-diff, y, :);
	        m = m + 1;
	    end;
	    if sum(img(x+diff, y, :)) <= threshold
	        neighbor_sum = neighbor_sum + img(x + diff, y, :);
	        m = m + 1;
	    end;
            if sum(img(x, y-diff, :)) <= threshold
	        neighbor_sum = neighbor_sum + img(x, y-diff, :);
	        m = m + 1;
	    end;
	    if sum(img(x, y+diff, :)) <= threshold
	        neighbor_sum = neighbor_sum + img(x, y+diff, :);
	        m = m + 1;
	    end;
	    if m > 0 
	        img(x, y, :) = neighbor_sum / m;
	    end;
        end;
    end;

    imwrite(uint8(img), strcat(OUTDIR, sprintf('%02d',counter), '.png'), 'png');
    counter = counter + 1;
end;


