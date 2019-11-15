function [shape, tex, tl] = coeffs_to_head(alpha, beta, dim, blend, model, msz)
    %Generate a random head, render it and export it to PLY file
    %tic;
    if (nargin == 4)
      [model, msz] = load_model();
    end;
    msz.n_shape_dim = dim;
    msz.n_tex_dim = dim;
    if blend == 1
       shape  = coef2object( alpha, model.shapeMU, model.shapePC, model.shapeEV, model.segMM, model.segMB );
       tex    = coef2object( beta,  model.texMU,   model.texPC,   model.texEV, model.segMM, model.segMB );
    else
       shape  = coef2object( alpha, model.shapeMU, model.shapePC, model.shapeEV );
       tex    = coef2object( beta,  model.texMU,   model.texPC,   model.texEV);
    end;
    tl = model.tl;
    %toc;

    % Save it in PLY format
    % plywrite(filename, shape, tex, model.tl );
    % 
    % 
    % % Generate versions with changed attributes
    % apply_attributes(alpha, beta)
    % 
    % % Generate a random head with different coefficients for the 4 segments
    % shape = coef2object( randn(msz.n_shape_dim, msz.n_seg), model.shapeMU, model.shapePC, model.shapeEV, model.segMM, model.segMB );
    % tex   = coef2object( randn(msz.n_tex_dim,   msz.n_seg), model.texMU,   model.texPC,   model.texEV,   model.segMM, model.segMB );
    % 
    % plywrite('rnd_seg_head.ply', shape, tex, model.tl );
    % display_face(shape, tex, model.tl, rp);
