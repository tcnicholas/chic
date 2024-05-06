"""
02.02.24
@tcnicholas
Defaults for coarse-graining methods.
"""

import numpy as np


def centroid_method1(self):
    pass


def centroid_method2(self):
    pass


def imidazolate_ring(self, precision=1e-8, skip_elements=None, **kwargs):
    """
    Searches for the core 5-membered ring of an imidazolate molecule (by finding
    the 5-membered ring that contains the two connecting nitrogen atoms), and
    takes the centroid of that (excluding all substiuents and hydrogen atoms).
    """
    
    bead_count = 1
    non_framework_count = -1
    for label, cluster in self._atomic_clusters.items():

        # gather the rings for the current cluster. note, we do not have to deal
        # with single-metal nodes separately because the get_rings method
        # defaults to giving the one atom if the cluster only has one atom. this
        # will break for general usage with MOF metal clusters.
        find_rings_kwargs = {
            'including': ['C', 'N'],
            'connective_including': True,
            'strong_rings': True
        }
        find_rings_kwargs.update(**kwargs)

        # check coordination of the cluster meets the requirements specified.
        if not (
            self._minimum_coordination_numbers[label[0]] <=
            cluster.coordination_number <=
            self._maximum_coordination_numbers[label[0]]
        ):
            centroid = [cluster.get_centroid(skip_elements=skip_elements)]
            cluster.assign_beads(
                [non_framework_count],
                *self._wrap_cart_coords_to_frac(
                    centroid, precision=precision
                ),
                {i: [0] for i in cluster.site_indices},
                internal_bead_bonds=[]
            )
            cluster.skip = True
            non_framework_count -= 1
            continue

        # we require one ring only.
        rings = cluster.find_rings(**find_rings_kwargs)
        assert len(rings) == 1, "Ambiguous assignment of ZIF bonding ring!"

        # hence get centroid of ring.
        ring_indices = [
            cluster._site_indices.index(x) for x in rings[0]['nodes']
        ]
        ring_centroid = [np.mean(cluster._cart_coords[ring_indices], axis=0)]

        # assign the values to the cluster.
        cluster.assign_beads(
            [bead_count],
            *self._wrap_cart_coords_to_frac(ring_centroid, precision=precision),
            {i: [0] for i in cluster.site_indices},
            internal_bead_bonds=[]
        )
        cluster.skip = False
        bead_count += 1


methods = {

    'centroid_method1': {
        'func': centroid_method1,
        'bead_type': 'single',
    },

    'centroid_method2': {
        'func': centroid_method2,
        'bead_type': 'single',
    },
    
    'imidazolate_ring': {
        'func': imidazolate_ring,
        'bead_type': 'single'
    }

}
