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
    fp.close()
    return header


def add_process(atoms, header, ctf, start_id, stop_id, q, p_id):
    p = multiprocessing.Process(target=multi_run, args=(atoms, header, ctf, start_id, stop_id, q, p_id))
    p.start()


def multi_run(atoms, header, ctf, start_id, stop_id, q, p_id):
    print('Start process: ' + str(p_id))
    res = order_parameter(atoms, header, ctf, start_id, stop_id)

    q.put(res)
    print('Exit process: ' + str(p_id))


def order_parameter(atoms, header, ctf, start_id, stop_id):
    cn_tab = []
    op_tab = []
    x, y, z, a_id = header.index('x'), header.index('y'), header.index('z'), header.index('id')
    
    # Copy of original array
    atoms_temp = atoms.copy()

    # loop over each atom
    for atom in atoms:
        # Use of atoms selected for this process
        if start_id < atom[a_id] <= stop_id:
            # Find atoms where lenght betwee central atom and neighbours: r <= cutof
            # and omitting an atom with the same id as the central atom
            r_tab = np.logical_and(
                np.sqrt(
                    np.array(
                        (atom[x]-atoms_temp[:, x])**2 + 
                        (atom[y]-atoms_temp[:, y])**2 + 
                        (atom[z]-atoms_temp[:, z])**2
                    )
                ) <= ctf, atoms_temp[:, a_id] != atom[a_id] 
            )

            # From atoms in radius of cutoff compute lenght
            cn_selected = np.sqrt(
                np.array(
                    (atom[x]-atoms_temp[r_tab][:, x])**2 + 
                    (atom[y]-atoms_temp[r_tab][:, y])**2 + 
                    (atom[z]-atoms_temp[r_tab][:, z])**2
                )
            )

            # Number of atoms in cutof range it is Coordination number 
            cn = len(cn_selected)

            # Sum of all neighbours lenght from central atom it is Order Parameter
            op = np.sum(cn_selected)

            # Add this values to lists
            cn_tab.append(int(cn))
            op_tab.append(op)

    # Add OP and CN from lists to arrays
    new_atoms_mask = np.logical_and(atoms[:, a_id] > start_id, atoms[:, a_id] <= stop_id)
    new_atoms = atoms[new_atoms_mask]
    op_tab = np.array(op_tab).reshape(-1, 1)
    cn_tab = np.array(cn_tab).reshape(-1, 1)
    new_atoms = np.concatenate((new_atoms, cn_tab), axis=1)
    new_atoms = np.concatenate((new_atoms, op_tab), axis=1)

    # Return updated array 
    return new_atoms


def join_results(results):
    new_atoms = results[0]
    for i in range(1, len(results)):
        new_atoms = np.concatenate((new_atoms, results[i]), axis=0)
    return new_atoms


def save_results(path, header, results):
    fp = open(path + '.op.csv', 'a')
    fp.write(','.join(header) + ',CN,OP\n')

    for atom in results:
        fp.write(
            ','.join(np.array2string(atom, precision=5, separator=' ')[1:-1].split()) + '\n'
        )
    fp.close()


def cut_structure(atoms, header, ranges):
    box = calc_box(atoms, header)
    x, y, z = header.index('x'), header.index('y'), header.index('z')
    if ranges[0][0] == 'n':
        ranges[0][0] = box['min_X']
    if ranges[0][1] == 'n':
        ranges[0][1] = box['max_X']
    if ranges[1][0] == 'n':
        ranges[1][0] = box['min_Y']
    if ranges[1][1] == 'n':
        ranges[1][1] = box['max_Y']
    if ranges[2][0] == 'n':
        ranges[2][0] = box['min_Z']
    if ranges[2][1] == 'n':
        ranges[2][1] = box['max_Z']

    maskx = np.logical_and(atoms[:, x] >= float(ranges[0][0]), atoms[:, x] <= float(ranges[0][1]))
    masky = np.logical_and(atoms[:, y] >= float(ranges[1][0]), atoms[:, y] <= float(ranges[1][1]))
    maskz = np.logical_and(atoms[:, z] >= float(ranges[2][0]), atoms[:, z] <= float(ranges[2][1]))
    atoms_cuted = atoms[maskx * masky * maskz]

    return atoms_cuted


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


def plotting(path, results, header, ranges):
    bounds = np.array(list(range(0, 14, 1)))
    # Triangulations of data
    triang = tri.Triangulation(results[:, header.index('x')], results[:, header.index('y')])

    fig1, fig2 = plt.subplots()

    # Use tricounter to create distribution map of CN
    tcf = fig2.tricontourf(triang, results[:, -2], 13, cmap='rainbow', levels=bounds)
    fig2.set_xlabel('Y')
    fig2.set_ylabel('X')

    # Color bar - legend
    cbar = fig1.colorbar(tcf, cax=None, ax=None, fraction=0.025)
    cbar.set_ticks(bounds + 0.5)
    cbar.set_ticklabels(bounds)
    cbar.ax.set_xlabel('CN')
    
    # Save picture
    plt.savefig(path[:-4] + '.OP-map.png', dpi=300)


def main():
    path = sys.argv[1]      # path to file with positions of atoms

    cutof = 2.9             # Range where is searching neighbours
    n_process = 4           # Number of processes
    # ranges = [[xlo, xhi], [ylo, yhi], [zlo, zhi]]
    ranges = [['n', 'n'], ['n', 'n'], [5, 25]]      # box dimensions to plotting, n' means to keep original dimensions

    start_time = time.time()
    
    atoms = load_dump(path)
    header = load_header(path)
    
    # compute how many atoms will used per one process
    atoms_per_process = math.ceil(len(atoms)/n_process)

    # Queue, function that will wait for results from processes
    q = multiprocessing.Queue()

    # Start processes
    for i in range(0, n_process, 1):
        add_process(atoms, header, cutof, i*atoms_per_process, (i+1)*atoms_per_process, q, i+1)

    # Catching results, list of arrays
    # The function ends when ALL processes are finished!
    results = []
    for i in range(n_process):
        results.append(q.get())

    # Compute time
    print('--- %s seconds ---' % (time.time() - start_time))

    # Join results from all processes into one arrary
    results = join_results(results)

    # Save results into *.csv file before cutting
    save_results(path, header, results)

    # Cut atoms from surface
    results = cut_structure(results, header, ranges)

    # Plot cutted results
    plotting(path, results, header, ranges)


if __name__ == "__main__":
    main()
