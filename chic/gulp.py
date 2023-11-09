"""
26.10.23
@tcnicholas
GULP file handling and program-calling functions.
"""

import os
import asyncio
import subprocess
from typing import Union
from pathlib import Path
from itertools import combinations
from IPython.display import display, HTML

import numpy as np

from .vector import compute_angle, random_vector
from .utils import delete_files_except_suffixes, rename_file


class GulpWriter:

    def __init__(self,
        parent_structure,
        name,
        k_bond = 10,
        k_angle = 3,
        sub_elem: str = 'H',
        rattle: float = None,
        rtol: float = 2.0,
        keywords: list = None,
        commands: list = None
    ):
        self._parent_structure = parent_structure
        self._name = name
        self._sub_elem = sub_elem
        self._kbond = k_bond
        self._kangle = k_angle
        self._bead_to_node = {}
        self._node_frac_coords = {}
        self._connections = None
        self._rattle = rattle
        self._rtol = rtol
        self._keywords = keywords
        self._commands = commands
        self._simplify_node_labels()
        self._connect_a_nodes()

    
    def write_file(self, filename: Union[str, Path]):
        """
        Write the GULP input file.
        """

        sections = [
            self._header(),
            self._cell(),
            self._coords(),
            self._connect(),
            self._harmonic_bonds(),
            self._harmonic_angles()
        ]
        content = "".join(sections)
        
        # make sure directory exists to write to and if not, create it.
        Path(filename).parent.mkdir(parents=True, exist_ok=True)

        # write content to file.
        with open(filename, 'w') as f:
            f.write(content)


    def _header(self):
        """
        Append header lines.
        """
        # start the file with the keywords.
        h = f'{" ".join(self._keywords)}\n\n'

        # add the title and rtol lines.
        h += f'title\n{self._name}\nend\n\n'
        h += f'rtol {self._rtol}\n\n'

        # add additional commands.
        if self._commands is not None:
            h += '\n'.join(self._commands) + '\n\n'

        return h


    def _cell(self):
        """
        Write cell parameters to string.
        """
        lengths = self._parent_structure.lattice.lengths
        angles = self._parent_structure.lattice.angles
        return 'cell\n{:<12.8f} {:<12.8f} {:<12.8f} '.format(*lengths) + \
            '{:<12.8f} {:<12.8f} {:<12.8f}\n\n'.format(*angles)
    

    def _coords(self):
        """
        Write coordinates to file.
        """

        # setup random number generator if rattling is required.
        rng = np.random.default_rng()

        coords = 'fractional\n'
        for label, unit in self._parent_structure._atomic_clusters.items():

            if label[0] != 'a':
                continue

            if unit.bead_ids[0] < 0:
                continue

            assert len(unit.bead_ids) == 1, 'Only single-bead method supported.'

            # get the bead-ID and the node-ID.
            bead_id = unit.bead_ids[0]
            node_id = self._bead_to_node[bead_id]

            # if required, get a random displacement vector of magnitude up to 
            # the rattle value.
            disp = np.zeros(3)
            if self._rattle is not None:
                disp += random_vector(
                    rng.random(), rng.random(), norm=True
                ) * self._rattle

            # get the coordinates.
            frac_coords = self._parent_structure.lattice.get_fractional_coords(
                unit._cart_coords[0] + disp
            )

            # store the node coordinates.
            self._node_frac_coords[node_id] = frac_coords

            # write to string.
            coords += '{:<6} core {:>12.8f} {:>12.8f} {:>12.8f}\n'.format(
                f'{self._sub_elem}{node_id}', *frac_coords
            )


        return coords + '\nspace\n1\n\n'
    

    def _connect(self):
        """
        Write node connections to file.
        """
        connect_str = ''
        for connection in self._connections:
            connect_str += 'connect {:<3} {:<3} {:<3} {:<3} {:<3}\n'.format(
                connection[0], connection[1], *connection[2]
            )

        return connect_str + '\n'
    

    def _harmonic_bonds(self):
        """
        """
        bond_str = 'harm bond\n'
        bond_lengths = self._compute_equilibrium_bond_lengths()
        num_bonds = 0
        for pair in {(n1,n2) for (n1,n2,img) in self._connections}:
            bond_str += '{:<6} {:<6} {:>12.6f} {:>12.6f}\n'.format(
                f'{self._sub_elem}{pair[0]}', 
                f'{self._sub_elem}{pair[1]}',
                self._kbond,
                bond_lengths[tuple(sorted(pair))],
            )
            num_bonds += 1
        print('Added {} harmonic bonds.'.format(num_bonds))

        return bond_str + '\n'
    

    def _harmonic_angles(self):
        """
        """
        angle_str = 'three bond\n'
        for (n1, n2, n3), theta in self._compute_equilibrium_angles().items():
            angle_str += '{:<6} {:<6} {:<6} {:>12.6f} {:>12.6f}\n'.format(
                f'{self._sub_elem}{n1}',
                f'{self._sub_elem}{n2}',
                f'{self._sub_elem}{n3}',
                self._kangle,
                theta
            )
        return angle_str + '\n\n'


    def _simplify_node_labels(self):
        """
        We will lose the nodes associated with B-type beads, so we can renumber
        the nodes to be consecutive integers.
        """
        count = 1
        for label, unit in self._parent_structure._atomic_clusters.items():
            if label[0] != 'a':
                continue
            if unit.bead_ids[0] < 0:
                continue
            self._bead_to_node[unit.bead_ids[0]] = count
            count += 1

    
    def _compute_equilibrium_bond_lengths(self):
        """
        """
        dist = {tuple(sorted([n1,n2])):[] for (n1,n2,img) in self._connections}
        for (n1, n2, img) in self._connections:
            f1 = np.array(self._node_frac_coords[n1])
            f2 = np.array(self._node_frac_coords[n2]) + img
            d = self._parent_structure.lattice.get_all_distances(f1, f2)
            dist[tuple(sorted([n1,n2]))].append(d[0,0])
        return {k:np.mean(v) for k,v in dist.items()}
    

    def _compute_equilibrium_angles(self):
        """
        """
        all_bonds = {n1:set() for (n1,n2,img) in self._connections}
        all_bonds.update({n2:set() for (n1,n2,img) in self._connections})
        for (n1,n2,img) in self._connections:
            all_bonds[n1] |= {(n2,img)}
            all_bonds[n2] |= {(n1,tuple(-np.array(img)))}
        
        angles = {}
        for centralAtom, peripheralAtoms in all_bonds.items():
            for (atom1, atom3) in combinations(peripheralAtoms, r=2):
                
                # get atom1, atom2, atom3 label.
                n = (centralAtom, *sorted([atom1[0], atom3[0]]))
                
                # get fractional coordinates.
                fcentral = np.array(self._node_frac_coords[centralAtom])
                f1 = np.array(self._node_frac_coords[atom1[0]]) + np.array(atom1[1])
                f3 = np.array(self._node_frac_coords[atom3[0]]) + np.array(atom3[1])
                
                # compute the angles.
                all_cart = self._parent_structure._lattice.get_cartesian_coords(
                    [fcentral, f1, f3]
                )
                v12 = all_cart[1,:] - all_cart[0,:]
                v13 = all_cart[2,:] - all_cart[0,:]
                v23 = all_cart[2,:] - all_cart[1,:]
                a = compute_angle(v12, v13) * 180 / np.pi
                
                if n in angles:
                    angles[n].append(a)
                else:
                    angles[n] = [a]

        return {k:np.mean(v) for k,v in angles.items()}

    
    def _connect_a_nodes(self):
        """
        Connect together A-type nodes in the net. This is done by finding the 
        connections between A-B-A nodes and therefore inferring the connections
        between A-A nodes.
        """

        bead_to_cluster = {}
        for label, unit in self._parent_structure._atomic_clusters.items():

            if unit.bead_ids[0] < 0:
                continue

            for bead in unit.bead_ids:
                bead_to_cluster[bead] = label

        connect = set()
        for label, unit in self._parent_structure._atomic_clusters.items():

            # ignore B-type linkers.
            if label[0] != 'a':
                continue

            if unit.bead_ids[0] < 0:
                continue
            
            # get this bead-ID to get the neighbours. assumes a single-bead method
            # was used.
            central_bead = unit.bead_ids[0]
            neighbours = self._parent_structure._bead_neighbour_list[central_bead]

            # loop over the neighbours.
            for neighbour in neighbours:
                
                # get the bead-ID of the neighbour and the image.
                b_cluster = bead_to_cluster[neighbour['bead_id']]
                b_img = np.array(neighbour['image'])

                # find other A-type nodes connected to this B-type node.
                b_neighbours = self._parent_structure._bead_neighbour_list[neighbour['bead_id']]
                for a_neighbour in b_neighbours:
                    
                    a_cluster = bead_to_cluster[a_neighbour['bead_id']]
                    a_img = np.array(a_neighbour['image'])

                    # get the image of the A-type node relative to the central 
                    # A-type node.
                    tot_img = (a_img + b_img).astype(int)

                    # don't want to connect to self.
                    if a_neighbour['bead_id'] == central_bead and np.all(tot_img==0):
                        continue

                    # sort nodes by labels and move first atom to [0,0,0] cell.
                    bonds = [
                        (self._bead_to_node[central_bead], np.zeros(3, dtype=int)),
                        (self._bead_to_node[a_neighbour['bead_id']], tot_img)
                    ]
                    bonds.sort(key=lambda x: x[0])
                    bonds = [list(x) for x in bonds]
                    trans = bonds[0][1].copy().astype(int)
                    bonds[0][1] -= trans
                    bonds[1][1] -= trans
                    connect |= {(bonds[0][0], bonds[1][0], tuple(bonds[1][1]))}

        self._connections = connect


async def run_gulp_on_file(
    path_to_gulp: str, 
    file_path: Path
) -> bool:
    """
    Change directory to the input file, run gulp, and return to the original 
    directory. Designed to be used in a Jupyter Notebook.

    Arguments:
        path_to_gulp: The path to the gulp executable.
        file_path: The path to the .gin file.

    Returns:
        True if gulp command was successful, False otherwise.
    """

    # Ensure that file_path is of type Path
    if not isinstance(file_path, Path):
        file_path = Path(file_path)

    # Check if the specified file exists and is a .gin file
    if not file_path.is_file():
        display(HTML("<div style='color: red;'>Error: The specified file does not exist.</div>"))
        return False
    if file_path.suffix != '.gin':
        display(HTML("<div style='color: red;'>Error: The specified file is not a .gin file.</div>"))
        return False

    # Save the current working directory to return back to it later
    original_cwd = Path.cwd()
    try:
        # Change to the directory containing the input file
        os.chdir(file_path.parent)
        
        # Provide initial feedback
        display(HTML("<div style='color: blue;'>Running gulp command...</div>"))

        # Run the gulp command asynchronously and capture its output
        process = await asyncio.create_subprocess_exec(
            path_to_gulp, file_path.stem, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        # Provide feedback based on the result of the gulp command
        if process.returncode == 0:
            display(HTML("<div style='color: green;'>Gulp command was successful.</div>"))
            return True
        else:
            display(HTML(f"<div style='color: red;'>Gulp command failed with error:<pre>{stderr.decode()}</pre></div>"))
            return False
    except asyncio.TimeoutError:
        display(HTML("<div style='color: red;'>Error: Gulp command timed out.</div>"))
    except Exception as e:
        display(HTML(f"<div style='color: red;'>An unexpected error occurred: {str(e)}</div>"))
    finally:
        # Return to the original working directory
        os.chdir(original_cwd)
    return False


async def process_structure(
    structure, 
    path_to_gulp: str,
    eg_dir: Union[str, Path],
    min_supercell: int = 2,
    max_supercell: int = 8,
    step_supercell: int = 2,
    name: str = 'formate',
    find_atomic_cluster_kwargs: dict = None,
    get_coarse_grained_net_kwargs: dict = None,
    **kwargs
) -> None:
    """
    Process a structure object by generating different supercells, finding the
    coarse-grained net, writing to gulp format, and running gulp.

    :param structure: The structure object to process.
    :param eg_dir: The directory where output will be written.
    :param name: The base name for output files and directories (default is 'formate').
    """

    eg_dir = Path(eg_dir)

    for supercell in [1] + list(range(
        min_supercell, 
        max_supercell+step_supercell, 
        step_supercell
    )):

        # Generate supercell and determine coarse-grained net
        structure_supercell = structure.replicate([supercell] * 3)

        # Might need to pass in some kwargs to find_atomic_clusters() and
        # get_coarse_grained_net() to get the desired result.
        structure_supercell.find_atomic_clusters(
            **(find_atomic_cluster_kwargs or {})
        )
        structure_supercell.get_coarse_grained_net(
            **(get_coarse_grained_net_kwargs or {})
        )

        # write the intermediate coarse-grained net to lammps data file.
        structure_supercell.net_to_lammps_data(
            eg_dir / f'{name}-cg-{supercell}.data'
        )

        # Write structure to gulp format
        sub_dir = eg_dir / f'{name}-{supercell}'
        sub_dir.mkdir(exist_ok=True)
        delete_files_except_suffixes(sub_dir, 'gin')

        # Write gulp input file
        supercell_label = "x".join([str(supercell)] * 3)
        output_gin = sub_dir / f'{name}-{supercell_label}.gin'
        structure_supercell.net_to_gulp_framework_nodes(
            output_gin,
            name=f'{name}-{supercell_label}',
            sub_elem='H',
            commands=[f'output lammps {name}'],
            **kwargs
        )

        # Run gulp on the input file
        result = await run_gulp_on_file(path_to_gulp, str(output_gin))
        if not result:
            display(HTML("<div style='color: red;'>Gulp command failed.</div>"))
            continue

        # Rename the default gulp lammps outputs
        rename_file(sub_dir / f'{name}.lmp', f'{name}-{supercell_label}.data')
        rename_file(sub_dir / f'{name}.inp', f'{name}-{supercell_label}.in')
