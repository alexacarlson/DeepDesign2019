import argparse
import glob
import os
import subprocess

import chainer
import cupy as cp
import neural_renderer
import numpy as np
import scipy.misc
import tqdm

import deep_dream_3d


def make_gif(working_directory, filename):
    # generate gif (need ImageMagick)
    options = '-delay 8 -loop 0 -layers optimize'
    subprocess.call('convert %s %s/_tmp_*.png %s' % (options, working_directory, filename), shell=True)
    for filename in glob.glob('%s/_tmp_*.png' % working_directory):
        os.remove(filename)


def run():
    # settings
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--filename_obj', type=str)
    parser.add_argument('-o', '--filename_output', type=str)
    parser.add_argument('-d', '--output_directory', type=str)
    parser.add_argument('-al', '--adam_lr', type=float, default=0.01)
    parser.add_argument('-ab1', '--adam_beta1', type=float, default=0.9)
    parser.add_argument('-bs', '--batch_size', type=int, default=4)
    parser.add_argument('-is', '--image_size', type=int, default=224)
    parser.add_argument('-ni', '--num_iteration', type=int, default=1000)
    parser.add_argument('-cd', '--camera_distance', type=float, default=2.5)
    parser.add_argument('-ib', '--init_bias', type=str, default='(0,0,0)')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()
    args.init_bias = tuple([float(v) for v in args.init_bias[1:-1].split(',')])

    # create output directory
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    # setup chainer
    chainer.cuda.get_device_from_id(args.gpu).use()
    cp.random.seed(0)
    np.random.seed(0)

    # setup scene & optimizer
    model = deep_dream_3d.DeepDreamModel(
        args.filename_obj,
        camera_distance=args.camera_distance,
        init_bias=args.init_bias,
        image_size=args.image_size)
    model.to_gpu()
    optimizer = neural_renderer.Adam(alpha=args.adam_lr, beta1=args.adam_beta1)
    optimizer.setup(model)

    # optimization
    loop = tqdm.tqdm(range(args.num_iteration))
    for _ in loop:
        optimizer.target.cleargrads()
        loss = model(args.batch_size)
        loss.backward()
        optimizer.update()
        loop.set_description('Optimizing. Loss %.4f' % loss.data)

    # save obj
    #model.textures_1 = chainer.functions.concat((model.mesh.textures, model.mesh.textures.transpose((0, 1, 4, 3, 2, 5))), axis=1)
    #model.textures_1 = chainer.functions.concat((model.mesh.textures, model.mesh.textures), axis=1)
    #obj_fn = args.filename_output.split('/')[-1].split('.')[0]
    #neural_renderer.save_obj('%s/%s.obj'% (args.output_directory,obj_fn), model.mesh.vertices, model.mesh.faces, chainer.functions.tanh(model.textures_1).array)
    
    import pdb
    #pdb.set_trace()
    vertices,faces,textures = model.mesh.get_batch(args.batch_size)
    ## fill back
    textures_1 = chainer.functions.concat((textures, textures.transpose((0, 1, 4, 3, 2, 5))), axis=1)
    faces_1 = chainer.functions.concat((faces, faces[:, :, ::-1]), axis=1).data
    #
    obj_fn = args.filename_output.split('/')[-1].split('.')[0]
    output_directory = os.path.split(args.filename_output)[0]#'/'.join(args.filename_output.split('/')[-3:-1])
    #neural_renderer.save_obj('%s/%s.obj'% (output_directory,obj_fn), model.mesh.vertices.array, model.mesh.faces, model.mesh.textures.array)
    neural_renderer.save_obj('%s/%s.obj'% (output_directory,obj_fn), vertices[0], faces[0], textures[0].array)
    #neural_renderer.save_obj('%s/%s.obj'% (output_directory,obj_fn), model.mesh.vertices[0], model.mesh.faces[0], chainer.functions.tanh(model.textures_1[0]).array)

    # draw object
    model.renderer.background_color = (1, 1, 1)
    loop = tqdm.tqdm(range(0, 360, 4))
    for num, azimuth in enumerate(loop):
        loop.set_description('Drawing')
        model.renderer.eye = neural_renderer.get_points_from_angles(2.732, 30, azimuth)
        images = model.renderer.render(*model.mesh.get_batch(1))
        image = images.data.get()[0].transpose((1, 2, 0))
        scipy.misc.toimage(image, cmin=0, cmax=1).save('%s/_tmp_%04d.png' % (args.output_directory, num))
    make_gif(args.output_directory, args.filename_output)


if __name__ == '__main__':
    run()
