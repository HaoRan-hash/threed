import pyvista as pv
from pyvista import themes
import numpy as np
import os

def read_files(dir_name):
    files = os.listdir(dir_name)
    files = [os.path.join(dir_name, file) for file in files]
    print(files)
    
    # files里面只有6列，前3列是pos，后3列是color
    pos_s = []
    colors = []
    for file in files:
        points = np.loadtxt(file, delimiter=',')
        pos_s.append(points[:, 0:3])
        colors.append(points[:, 3:])
    
    return pos_s, colors


def vis_point_clouds(pos_s, colors):
    """
    pos_s: [(n, 3), ...]
    colors: [(n, 3), ...]
    """
    # 设置绘图主题
    my_theme = themes.DefaultTheme()
    my_theme.color = 'black'
    my_theme.lighting = True
    my_theme.show_edges = True
    my_theme.edge_color = 'white'
    my_theme.background = 'white'
    pv.set_plot_theme(my_theme)
    
    n_clouds = len(pos_s)
    plotter = pv.Plotter(shape=(1, n_clouds), border=False)
    
    for i in range(n_clouds):
        plotter.subplot(0, i)
        plotter.add_points(pos_s[i], scalars=colors[i], rgb=True, opacity=1.0, point_size=5.0, render_points_as_spheres=True)
    plotter.link_views()   # 同步所有子图视角
    
    plotter.show()
    plotter.close()


if __name__ == '__main__':
    pos_s, colors = read_files('10')
    vis_point_clouds(pos_s, colors)
