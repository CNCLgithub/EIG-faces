function handle = display_intrinsics (shp_ground, shp, tex, tl, rp, mode_az,mode_ev, particle_id)

    shp = reshape(shp, [ 3 prod(size(shp))/3 ])'; 
    shp_ground = reshape(shp, [ 3 prod(size(shp_ground))/3 ])'; 
    tex = reshape(tex, [ 3 prod(size(tex))/3 ])'; 
    tex = min(tex, 255);
    
    if isequal(particle_id, []) == 1
        particle_id = 1;
    end
    
    handle = figure(particle_id);
    set(handle,'Visible','off');
    set(handle, 'PaperPositionMode','auto');
    
    set(gcf, 'Renderer', 'opengl');
    
    fig_pos = get(handle, 'Position');
	
    fig_pos(3) = rp.width;
    fig_pos(4) = rp.height;
    set(handle, 'Position', fig_pos);
    set(handle, 'ResizeFcn', @resizeCallback);

    %shift camera
    mesh_h = trimesh(...
		      tl, shp_ground(:, 1), shp_ground(:, 3), shp_ground(:, 2), ...
    		     'EdgeColor', [1 1 1], ...
		     'FaceColor', [1 1 1], ...
		     'FaceLighting', 'none', ...
		     'FaceAlpha', 0, ...
		     'EdgeAlpha', 0 ...
		    );
    view(180, 0);
    hold on;

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
    
    material([.5, .5, .1 1  ])

    camlight(mode_az,mode_ev);
	

%% ------------------------------------------------------------CALLBACK--------
function resizeCallback (obj, eventdata)
	
	fig = gcbf;
	fig_pos = get(fig, 'Position');

	axis = findobj(get(fig, 'Children'), 'Tag', 'Axis.Head');
	set(axis, 'Position', [ 0 0 fig_pos(3) fig_pos(4) ]);
	
