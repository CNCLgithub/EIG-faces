baselpath = '/gpfs/milgram/project/yildirim/ilker/datasets/PublicMM1';
baselmatlabpath = '/gpfs/milgram/project/yildirim/ilker/datasets/PublicMM1/matlab';

addpath(baselpath);
addpath(baselmatlabpath);

[model, msz] = load_model();

load('../../bfm09-generator/mask_faces.mat');
mask = tr;
mask_tl = model.tl(mask == 1,:);
mask_tl = reshape(mask_tl, [57776*3, 1]);
dim = 50;

%inputmodel = 'eig';
inputmodel = 'eig_classifier';

if strcmp(inputmodel, 'eig') == 1
  inputfile = './output/eig.hdf5';
  outputfile = './output/eig_predicted_depth.txt';
else
  inputfile = './output/eig_classifier.hdf5';
  outputfile = './output/eig_classifier_predicted_depth.txt';
end

output = hdf5read(inputfile, '0'); 
output = transpose(output);
N = size(output, 1);
disp(N);
results = zeros(1, N);
for i = 1:N
  latents = output(i, :);
  alpha = latents(1:200);
  beta = latents(201:400);
  alpha = reshape(alpha, [dim,4]);
  beta = reshape(beta, [dim,4]);
  [shape, tex, back_tl] = coeffs_to_head(alpha, beta, dim, 1, model, msz);
  M = length(shape);
  z_vals = shape(3:3:M);

  % nose, leftcheek, rightcheek
  results(i) = (z_vals(8320) + z_vals(12061) + z_vals(4065))/3.;

end;

dlmwrite(outputfile, results);



