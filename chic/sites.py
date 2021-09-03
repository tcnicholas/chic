"""
01.06.21
@tcnicholas
Sort elements into different sites.

Use a simple sorting algorithm for classifying elements into site "types". The
algorithm was devised with MOFs in mind, and therefore there are many instances
where it will fall short; users can, however, specify the sites explicitly.

New classification algorithms could also be implemented to deal with other types
of system, simply by defining a new function like "mof()", and calling it from
the sort_sites() using frameworkType = <my_new_algorithm>.

#TODO: might be interesting if one could train a machine-learning classification
model to predict whether sites would be A or B sites.
"""


import re


InCHI =  {
    "nonMetals" :   ["H", "He", "B", "C",  "N",  "O",  "F", "Ne",
                                    "Si",  "P",  "S", "Cl", "Ar",
                                    "Ge", "As", "Se", "Br", "Kr",
                                                "Te",  "I", "Xe",
                                                      "At", "Rn",
                    ],

    "nonMetalExceptions" : ["B", "Si","P"] # Found in practice to be otherwise.
}

one_cn = ["H","F","Cl"]


def sort_sites(structure, method):
    """
    Sorts all elements in 'structure' into different site "types", e.g. A sites,
    B sites, etc. according to the method specified.

    Args
    ----
    structure: pymatgen.core.Structure
        Pymatgen Structure object.

    method: str   
        name of sort_sites() algorithm to use (in sites.py). New custom
        methods should be added to the site.py module so that they are
        callable from the main Structure module.
                    
        (i) "mof": developed for work in Ref 1. Works well for a wide range
            of AB2 MOF structures.

        (ii) "allAtoms": makes each element in structure into its own
            site-type class. e.g. for Tb(HCO2)3, the resultant sites would
            be: 
            
            [["Tb"], ["H"], ["C"], ["O"]] 
            
            (not necessarily in that order).
    """

    # Get elements.
    e = structure.composition.elements

    return eval(f"{method}(structure, e)")

 
def allAtoms(structure, elements):
    """
    Separate all elements into different sites.

    Args
    ----
    structure: pymatgen.core.Structure
        Pymatgen Structure object.

    elements:
    """
    return [[e.symbol] for e in elements]


def mof(structure, elements):
    """
    Sort sites into A and B sites. Designed with AB2 MOF frameworks in mind. The
    general idea is to distinguish "metal" centres from organic linkers.

    Args
    ----
    structure: pymatgen.core.Structure
        Pymatgen Structure object.

    elements:
    """

    def get_type(structure, e):
        """
        For a given element, decide if it is most likely to be a node or linker.

        Args
        ----
        structure: pymatgen.core.Structure
            Pymatgen Structure object.

        e:
        """
        # --------------------- #
        # (1) TRANSITION METALS #
        # --------------------- #
        # Assign all transition metals as nodes. There are some instances where
        # this will fail (e.g. **insert W example structure).
        if e.is_transition_metal:
            return "a"
        
        # -------------------- #
        # (2) InCHI non-metals #
        # -------------------- #
        # Assign InCHI classified "non-metals" to linker, with a few exceptions:
        #    (A) ices/clathrates have oxygen as a node (OH2);
        #    (B) B, Si, P can also be nodes (BIFs, silica, phosphates etc.);
        elif e.symbol in InCHI["nonMetals"]:

            if e.symbol == "O" and is_ice(structure):
                #TODO: identifty clathrates too. These will often have organic
                # molecules in the pores that need to be "removed".
                return "a"
            
            elif e.symbol in InCHI["nonMetalExceptions"]:

                #TODO: determine local coordination. If not extended beyond a
                # small molecule, throw away.
                return "a"

            else:
                return "b"
        
        # -------------------- #
        # (3) Group 1/2 metals #
        # -------------------- #
        # With the exceptions of Li and Be, throw away alkali and alkaline earth
        # metals. Li and Be are often nodes.
        if e.is_alkali or e.is_alkaline and e.symbol not in ["Li", "Be"]:
            return None
        else:
            return "a"
        
        return "a"
    
    # For all elements, get site-type.
    atom_site = {"a":[], "b":[]}
    for e in elements:
        atom_site[get_type(structure, e)].append(e.symbol)

    # Print sorted sites to terminal.
    print(f"* sites * A sites: {', '.join(atom_site['a'])}")
    print(f"* sites * B sites: {', '.join(atom_site['b'])}")

    return list(atom_site.values())


def is_ice(structure):
    """
    Returns True if pure ice is suspected.

    Args
    ----
    structure: pymatgen.core.Structure
        Pymatgen Structure object.
    """
    if re.sub(" ","",structure.composition.alphabetical_formula) == "H2O":
        return True
    return False


