function handle = display_intrinsics (shp, tex, tl, rp, mode_az,mode_ev, particle_id, mask, shift_camera)

    shp = reshape(shp, [ 3 prod(size(shp))/3 ])'; 
    tex = reshape(tex, [ 3 prod(size(tex))/3 ])'; 
    tex = min(tex, 255);
    
    if isequal(particle_id, []) == 1
        particle_id = 1;
    end
    
    handle = figure(particle_id);
    set(handle,'Visible','off');
    set(handle, 'PaperPositionMode','auto');
    
    set(gcf, 'Renderer', 'opengl');
    %set(handle, 'Renderer', 'zbuffer');
    
    fig_pos = get(handle, 'Position');
	
    fig_pos(3) = rp.width;
    fig_pos(4) = rp.height;
    set(handle, 'Position', fig_pos);
    set(handle, 'ResizeFcn', @resizeCallback);


    if nargin < 9
      shift_camera = 0; %if 1, then first load canonical face and align the camera wrt it.
    end;
    if particle_id == 1
	if shift_camera == 1
          mesh_h = trimesh(...
		tl, shp(:, 1), shp(:, 3), shp(:, 2), ...
    		'EdgeColor', [1 1 1], ...
		'FaceColor', [1 1 1], ...
		'FaceLighting', 'none', ...
		'FaceAlpha', 0, ...
		'EdgeAlpha', 0 ...
	  );
	  view(180, 0);
	  hold on;
	  tl = tl(mask == 1,:);
	end;

	RotAngle = roty(rp.phi);
	tmp = RotAngle * shp';
	shp = tmp';
    
	RotAngle = rotx(rp.elevation);
	tmp = RotAngle * shp';
	shp = tmp';

       % regular
        mesh_h = trimesh(...
	    tl, shp(:, 1), shp(:, 3), shp(:, 2), ...
    	    'EdgeColor', 'none', ...
	    'FaceVertexCData', tex/255, 'FaceColor', 'interp', ...
	    'FaceLighting', 'phong' ...
	);
	hold off;
    elseif particle_id == 2
	% albedo
	if shift_camera == 1
          mesh_h = trimesh(...
		tl, shp(:, 1), shp(:, 3), shp(:, 2), ...
    		'EdgeColor', [1 1 1], ...
		'FaceColor', [1 1 1], ...
		'FaceLighting', 'none', ...
		'FaceAlpha', 0, ...
		'EdgeAlpha', 0 ...
	  );
	  view(180, 0);
	  hold on;
	  tl = tl(mask == 1,:);
	end;
	RotAngle = roty(rp.phi);
	tmp = RotAngle * shp';
	shp = tmp';
    
	RotAngle = rotx(rp.elevation);
	tmp = RotAngle * shp';
	shp = tmp';

	% albedo
        mesh_h = trimesh(...
	    tl, shp(:, 1), shp(:, 3), shp(:, 2), ...
    	    'EdgeColor', 'none', ...
	    'FaceVertexCData', tex/255, 'FaceColor', 'interp', ...
	    'FaceLighting', 'none' ...
	);
	if shift_camera == 1
	  hold off;
	else
	  view(180, 0);
	end
    elseif particle_id == 3
	% normals
        if shift_camera == 1
          mesh_h = trimesh(...
	    tl, shp(:, 1), shp(:, 3), shp(:, 2), ...
    	    'EdgeColor', [1 1 1], ...
	    'FaceColor', [1 1 1], ...
	    'FaceLighting', 'none', ...
	    'FaceAlpha', 0, ...
	    'EdgeAlpha', 0 ...
	  );
	  view(180, 0);
	  hold on;
	  tl = tl(mask == 1,:);
        end;

	RotAngle = roty(rp.phi);
	tmp = RotAngle * shp';
	shp = tmp';
    
	RotAngle = rotx(rp.elevation);
	tmp = RotAngle * shp';
	shp = tmp';

	% normals
        tr = triangulation(tl, shp(:, 1), shp(:, 3), shp(:, 2));
        vn = vertexNormal(tr);
        vn = ((vn + 1)./2).*255;
        %vn = [vn(:,3) vn(:,1) vn(:,2)];
        mesh_h = trimesh(...
            tl, shp(:, 1), shp(:, 3), shp(:, 2), ...
            'EdgeColor', 'none', ...
            'FaceVertexCData', vn/255, 'FaceColor', 'interp', ...
            'FaceLighting', 'none', ...
            'AmbientStrength', 1.0 ...
        );
	if shift_camera == 1
	  hold off;
	else
	  view(180, 0);
	end
    end;

    set(gca, ...
	'DataAspectRatio', [ 1 1 1 ], ...
	'PlotBoxAspectRatio', [ 1 1 1 ], ...
	'Units', 'pixels', ...
	'GridLineStyle', 'none', ...
	'Position', [ 0 0 fig_pos(3) fig_pos(4) ], ...
	'Visible', 'off', 'box', 'off', ...
	'Projection', 'perspective' ...
    ); 
	
    set(handle, 'Color', [ 1 1 1 ]); 
    
    %view(180 + rad2deg(rp.phi), rad2deg(rp.elevation));
    %view(180,0);

    material([.5, .5, .1 1  ])

    camlight(mode_az,mode_ev);
	

%% ------------------------------------------------------------CALLBACK--------
function resizeCallback (obj, eventdata)
	
	fig = gcbf;
	fig_pos = get(fig, 'Position');

	axis = findobj(get(fig, 'Children'), 'Tag', 'Axis.Head');
	set(axis, 'Position', [ 0 0 fig_pos(3) fig_pos(4) ]);
	
