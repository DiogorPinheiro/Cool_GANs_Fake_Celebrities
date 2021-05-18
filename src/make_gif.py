import glob
import imageio
import tensorflow_docs.vis.embed as embed

def make_gif(gif_name):
    anim_file = gif_name + '.gif'

    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob('assets/20210515-215159-faces/epoch*.png')
        filenames = sorted(filenames)
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)

    embed.embed_file(anim_file)

if __name__ == '__main__':
    make_gif('dcgan_faces_369')