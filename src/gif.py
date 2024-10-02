import os
import imageio


def gif_folder(folder, prefix, fps=20):
    m = 0
    images = []
    while True:
        png_name = f'{folder}/{prefix}{m}.png'
        if not os.path.exists(png_name):
            break
        images.append(imageio.imread(png_name))
        m += 1
    # imageio.mimsave(f'{folder}/{prefix}.gif', images, fps=fps, loop=0)
    imageio.mimsave(f'{prefix}.gif', images, fps=fps, loop=0)
