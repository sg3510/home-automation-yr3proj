% load acc3d
% scatter3(acc_3d(:,1)/60,acc_3d(:,2),acc_3d(:,3),'CData',acc_3d(:,3))
% plot3(acc_3d(:,1)/60,acc_3d(:,2),acc_3d(:,3),'-.')
tri = delaunay(acc_3d(:,1)/12*ratio,acc_3d(:,2)/12*ratio);
h = trisurf(tri, acc_3d(:,1)/12*ratio,acc_3d(:,2)/12*ratio,acc_3d(:,3));
axis vis3d

xlabel('Sample p (hours)')
ylabel('Sample f (hours)')
zlabel('Accuracy (%)')
l = light('Position',[100 50 150])
set(gca,'CameraPosition',[208 -50 200])
% lighting phong
% shading interp
% colorbar EastOutside
