import math
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import multiprocessing
import sys
import time


def load_dump(path):
    atoms = np.genfromtxt(path, skip_header=9)
    return atoms


def load_header(path):
    fp = open(path)
    header = ''
    for i, line in enumerate(fp):
        if i == 8:
            header = line.strip().split()[2::]
            break
    return header


def calc_box(atoms, header):
    box = {}
    maxs = np.amax(atoms, axis=0)
    mins = np.amin(atoms, axis=0)
    box['max_X'] = maxs[header.index('x')]
    box['min_X'] = mins[header.index('x')]
    box['len_X'] = box['max_X'] - box['min_X']
    box['max_Y'] = maxs[header.index('y')]
    box['min_Y'] = mins[header.index('y')]
    box['len_Y'] = box['max_Y'] - box['min_Y']
    box['max_Z'] = maxs[header.index('z')]
    box['min_Z'] = mins[header.index('z')]
    box['len_Z'] = box['max_Z'] - box['min_Z']
    return box


def cut_structure(atoms, header, range):
    box = calc_box(atoms, header)
    if range[0][0] == 'n':
        range[0][0] = box['min_X']
    if range[0][1] == 'n':
        range[0][1] = box['max_X']
    if range[1][0] == 'n':
        range[1][0] = box['min_Y']
    if range[1][1] == 'n':
        range[1][1] = box['max_Y']
    if range[2][0] == 'n':
        range[2][0] = box['min_Z']
    if range[2][1] == 'n':
        range[2][1] = box['max_Z']

    maskx = np.logical_and(atoms[:, 2] >= float(range[0][0]), atoms[:, 2] <= float(range[0][1]))
    masky = np.logical_and(atoms[:, 3] >= float(range[1][0]), atoms[:, 3] <= float(range[1][1]))
    maskz = np.logical_and(atoms[:, 4] >= float(range[2][0]), atoms[:, 4] <= float(range[2][1]))
    atoms_cuted = atoms[maskx * masky * maskz]

    return atoms_cuted


def order_parameter(atoms, header, ctf, start_id, stop_id):
    cn_tab = []
    op_tab = []
    # id_tab = []
    x, y, z, a_id = header.index('x'), header.index('y'), header.index('z'), header.index('id')
    
    atoms_temp = atoms.copy()

    for atom in atoms:
        if start_id < atom[a_id] <= stop_id:
            r_tab = np.logical_and(
                np.sqrt(
                    np.array(
                        (atom[x]-atoms_temp[:, x])**2 + 
                        (atom[y]-atoms_temp[:, y])**2 + 
                        (atom[z]-atoms_temp[:, z])**2
                    )
                ) <= ctf, atoms_temp[:, a_id] != atom[a_id]
            )
            cn_selected = np.sqrt(
                np.array(
                    (atom[x]-atoms_temp[r_tab][:, x])**2 + 
                    (atom[y]-atoms_temp[r_tab][:, y])**2 + 
                    (atom[z]-atoms_temp[r_tab][:, z])**2
                )
            )
            cn = len(cn_selected)
            op = np.sum(cn_selected)

            cn_tab.append(int(cn))
            op_tab.append(op)
            # id_tab.append(int(atom[a_id]))

    new_atoms_mask = np.logical_and(atoms[:, a_id] > start_id, atoms[:, a_id] <= stop_id)
    new_atoms = atoms[new_atoms_mask]
    op_tab = np.array(op_tab).reshape(-1, 1)
    cn_tab = np.array(cn_tab).reshape(-1, 1)
    # id_tab = np.array(id_tab).reshape(-1, 1)
    new_atoms = np.append(new_atoms, cn_tab, axis=1)
    new_atoms = np.append(new_atoms, op_tab, axis=1)
    # new_atoms = np.append(new_atoms, id_tab, axis=1)

    return new_atoms


def add_process(atoms, header, ctf, start_id, stop_id, q, p_id):
    p = multiprocessing.Process(target=multi_run, args=(atoms, header, ctf, start_id, stop_id, q, p_id))
    p.start()


def multi_run(atoms, header, ctf, start_id, stop_id, q, p_id):
    print('Start process: ' + str(p_id))
    res = order_parameter(atoms, header, ctf, start_id, stop_id)

    q.put(res)
    print('Exit process: ' + str(p_id))


def save_results(path, header, atoms, res):
    fp = open(path + '.op.csv', 'a')
    fp.write(','.join(header) + ',CN,OP\n')

    for elr in res:
        line = atoms[atoms[:, header.index('id')] == elr[0]]
        fp.write(
            ','.join(np.array2string(line[0], precision=5, separator=' ')[1:-1].split()) + 
            ',' + str(elr[1]) + ',' + str(elr[2]) + '\n'
        )
    fp.close()


def plotting(path, results, header, ranges):
    bounds = np.array(list(range(0, 14, 1))) 
    triang = tri.Triangulation(results[:, header.index('x')], results[:, header.index('y')])

    fig1, ax1 = plt.subplots()
    tcf = ax1.tricontourf(triang, results[:, -2], 13, cmap='rainbow', levels=bounds)
    cbar = fig1.colorbar(tcf, cax=None, ax=None, fraction=0.025)
    cbar.set_ticks(bounds + 0.5)
    cbar.set_ticklabels(bounds)
    cbar.ax.set_xlabel('CN')

    plt.xlabel('Y')
    plt.ylabel('X')
    
    plt.savefig(path[:-4] + '.OP-map.png', dpi=300)

def join_results(results):
    new_atoms = results[0]
    for i in range(0, len(results)):
        if i != 0:
            new_atoms = np.append(new_atoms, results[i], axis=0)
    return new_atoms


def main():
    path = sys.argv[1]

    cutof = 2.9             # Range where is searching neighbours
    n_process = 4
    # ranges = [[xlo, xhi], [ylo, yhi], [zlo, zhi]]
    ranges = [['n', 'n'], ['n', 'n'], [5, 25]]

    start_time = time.time()
    
    atoms = load_dump(path)
    header = load_header(path)
    
    atoms_per_process = math.ceil(len(atoms)/n_process)
    q = multiprocessing.Queue()

    for i in range(0, n_process, 1):
        add_process(atoms, header, cutof, i*atoms_per_process, (i+1)*atoms_per_process, q, i+1)

    results = []
    for i in range(n_process):
        results.append(q.get())

    print('--- %s seconds ---' % (time.time() - start_time))

    results = join_results(results)
    plotting(path, results, header, ranges)


if __name__ == "__main__":
    main()
