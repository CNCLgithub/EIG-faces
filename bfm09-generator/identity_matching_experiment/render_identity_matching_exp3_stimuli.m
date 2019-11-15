% Render Identity Matching Experiment 1 stimuli. First 96 pairs are test stimuli; remaining 10 pairs are for the training portion of the experiment.

ROOT = getenv('PYTHONPATH'); % ROOT of the project folder
ROOT = split(ROOT, ':');
ROOT = ROOT{end};

baselpath = strcat(ROOT, '/bfm09-generator', '/bfm_utils/PublicMM1');
baselmatlabpath = strcat(baselpath, '/matlab');
addpath(baselpath);
addpath(baselmatlabpath);

OUTDIR = strcat('./output/IM-exp3/');
mkdir(OUTDIR);

load('identity_matching_experiments_identities.mat');
coeffs = all_latents;
load('identity_matching_experiments_scene_parameters.mat');

[model, msz] = load_model();
DIM = 50; % length of dimension per part -- total dimensionality 50*4=200 for shape and 50*4=200 for texture.
% set general rendering parameters
rp = defrp;
rp.dir_light.dir = [0;1;1];
rp.dir_light.intens = .6.*ones(3,1);
rp.width = 224;
rp.height=224;

counter = 1;
for i=1:212
    turn = mod(i-1,2)+1;
    % Is this a same or different trial?
    if (i <= 96 && turn == 2) || (i > 192 && i <=202 && turn == 2)
        shape = coeffs(i-1,1:200);
        texture = coeffs(i-1,201:400);
    else
        shape = coeffs(i,1:200);
        texture = coeffs(i,201:400);
    end

    alpha = reshape(shape, [DIM,4]);
    beta = reshape(texture, [DIM,4]);
    [shape, tex, tl] = coeffs_to_head(alpha, beta, DIM, 1);

    if turn == 1
        %study item, regularly rendered.
        rp.phi = view_list(i,1);
        rp.elevation = view_list(i,2);
        rp.mode_az = view_list(i,3);
        rp.mode_el = view_list(i,4);
        handle = display_face(shape, tex, tl, rp, rp.mode_az, rp.mode_el, []);
        img = hardcopy(handle, '-dopengl', '-r300');
        img = imresize(img, [224, 224]);
    else
        % test item; rendered with fish-eye lens distortion and with frontal viewing
	rp.phi = 0; 
        rp.elevation = 0; 
        rp.mode_az = 0; 
        rp.mode_el = 0;
        handle = display_face(shape, tex, tl, rp, rp.mode_az, rp.mode_el, 3);
        img = hardcopy(handle, '-dopengl', '-r300');
        img = imresize(img, [224, 224]);

        options = [224 224 1.35];
        tf = maketform('custom', 2, 2, [], @fisheye_inverse, options);
        img = imtransform(img, tf);
        img(img == 0) = 255;
        img = imresize(img, [224, 224]);

        % the fish-eye distortion function introduces artifacts in the image which we clean up below using some rather ugly code.
        img(1:2,:,:) = 255; img(:,1:2,:) = 255;
        img(223:224,:,:) = 255; img(:, 223:224, :) = 255;

        t = sum(img,3);
        for m = 3:222
	    for n = 3:222
	        ss = 0;
                if t(m,n) == 765
                    continue;
                else
		  if t(m-2,n) == 765 ss = ss+1; end; if t(m+2,n) == 765 ss = ss+1; end; if t(m, n-2) == 765 ss = ss+1; end; if t(m, n+2) == 765 ss = ss+1;end;
                  if ss >= 3
                     img(m,n,:) = 255;
                  end;
                end;
             end;
        end;

        t = sum(img,3);
        for m = 3:222
	    for n = 3:222
	        ss = 0;
                if t(m,n) == 765
                    continue;
                else
		  if t(m-1,n) == 765 ss = ss+1; end; if t(m+1,n) == 765 ss = ss+1; end; if t(m, n-1) == 765 ss = ss+1; end; if t(m, n+1) == 765 ss = ss+1;end;
                  if ss >= 2
                     img(m,n,:) = 255;
                  end;
                end;
             end;
        end;

        t = sum(img,3);
        for m = 3:222
	    for n = 3:222
	        ss = 0;
                if t(m,n) == 765
                    continue;
                else
		  if t(m-1,n) == 765 ss = ss+1; end; if t(m+1,n) == 765 ss = ss+1; end; if t(m, n-1) == 765 ss = ss+1; end; if t(m, n+1) == 765 ss = ss+1;end;
                  if ss >= 2
                     img(m,n,:) = 255;
                  end;
                end;
             end;
        end;
    end;

    imwrite(img, strcat(OUTDIR, int2str(ceil(i/2.)), '_', int2str(turn), '.png'));
    disp(counter);
    counter = counter + 1;
end;
