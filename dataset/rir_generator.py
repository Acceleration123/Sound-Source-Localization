import argparse
import pyroomacoustics as pra
import numpy as np
import random
from tqdm import tqdm
import math
from itertools import product
import os


# rir for linear array
def get_pos(room_dim, d_from_wall, array_num, d_inter):
    """
    output:
      array position: 3 x array_num
      source position: 3 x 2
    """
    length, width, height = room_dim
    R = [0.5, 1, 2, 3]

    flag = 0
    while flag == 0:
        # speaker and noise source
        r1 = R[random.randint(0, len(R) - 1)]
        phi1 = random.uniform(-1 / 2, 1 / 2) * math.pi

        # limit the position of the array center
        d_right = max(d_from_wall, abs(r1 * np.sin(phi1)))
        d_left = max(d_from_wall, abs(r1 * np.sin(phi1)))
        d_front = max(d_from_wall, abs(r1 * np.cos(phi1)))
        d_back = d_from_wall

        # array position
        # x_center and y_center may be negative!
        x_center = d_left + (length - d_left - d_right) * random.uniform(0, 1)
        y_center = d_front + (width - d_front - d_back) * random.uniform(0, 1)
        z_center = 1 + random.uniform(0, 1)
        array_pos = np.array([[x_center + (k - (array_num - 1) / 2) * d_inter, y_center, z_center]
                              for k in range(array_num)]).T  # (3, array_num)

        # source position
        source_pos = (np.array([[x_center, y_center, z_center]]).T +
                      np.array([[r1 * np.sin(phi1), r1 * np.cos(phi1), 0]]).T)  # (3, 1)
        params = [r1, phi1]

        if np.all((array_pos[0, :] > 0) & (source_pos[0, :] > 0) & (source_pos[0, :] < length) &
                  (array_pos[1, :] > 0) & (source_pos[1, :] > 0) & (source_pos[1, :] < width)):
            flag = 1

    return array_pos, source_pos, params


def rir_generator(room_dim, rt_60, array_pos, source_pos, fs, direct=False):

    e_absorption, max_order = pra.inverse_sabine(rt_60, room_dim)
    room = pra.ShoeBox(room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=0 if direct else max_order)

    room.add_source(source_pos[:, 0])
    room.add_microphone_array(pra.MicrophoneArray(array_pos, fs))

    room.compute_rir()

    return room.rir


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-num', '--num_of_rir', type=int, default=5000, help='number of rir to generate')
    parser.add_argument('-path', '--rir_save_path', help='path to save rir in reverberant environment')
    args = parser.parse_args()

    for i in tqdm(range(args.num_of_rir)):
        # simulation parameters initialization (randomly)
        # room: (length, width, height)
        room_dimension = [random.uniform(3, 10), random.uniform(3, 10), random.uniform(2.5, 3)]

        # reverberation time
        l, w, h = room_dimension
        r = 0.163 * np.prod(room_dimension) / (2 * (l * w + l * h + w * h))
        r = max(r, 0.05)
        room_rt60 = random.uniform(r, 1.0)

        # source position and array center position
        d_inter = 0.02
        array_position, source_position, params = get_pos(room_dimension, d_from_wall=0.5, array_num=4, d_inter=d_inter)

        rir_direct = rir_generator(room_dimension, room_rt60, array_position, source_position, fs=16000, direct=True)
        rir_reverb = rir_generator(room_dimension, room_rt60, array_position, source_position, fs=16000, direct=False)

        # save the rir
        min_len_direct = min(len(rir_direct[0][0]), len(rir_direct[1][0]), len(rir_direct[2][0]), len(rir_direct[3][0]))
        for m, n in product(range(4), range(1)):
            rir_direct[m][n] = rir_direct[m][n][:min_len_direct]

        min_len_reverb = min(len(rir_reverb[0][0]), len(rir_reverb[1][0]), len(rir_reverb[2][0]), len(rir_reverb[3][0]))
        for m, n in product(range(4), range(1)):
            rir_reverb[m][n] = rir_reverb[m][n][:min_len_reverb]

        rir_direct = np.array(rir_direct)
        rir_reverb = np.array(rir_reverb)

        save_name = f'{str(i).zfill(4)}_rt60={room_rt60:.2f}_d={d_inter:.3f}_r1={params[0]:.2f}_phi1={(params[1]/math.pi*180):.2f}.npy'
        save_path_direct = os.path.join(args.rir_save_path.replace('reverb', 'direct'), save_name)
        save_path_reverb = os.path.join(args.rir_save_path, save_name)

        np.save(save_path_direct, rir_direct)
        np.save(save_path_reverb, rir_reverb)



