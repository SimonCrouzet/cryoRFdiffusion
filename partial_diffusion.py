import argparse
from itertools import groupby
import json
import logging
from operator import itemgetter
from pathlib import Path, PosixPath
import os, sys, subprocess
from typing import List, Tuple, Dict, Optional, Union
from Bio import PDB
from collections import defaultdict
import ast

def load_cdr_length_data(json_file_path: Optional[str], data_type: str = 'all') -> Optional[Dict]:
    if json_file_path:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        if data_type not in ['all', 'train']:
            logging.warning(f"Invalid data_type '{data_type}'. Using 'all' instead.")
            data_type = 'all_data'
        return data[data_type + '_data']
    return None

def parse_pdb(pdb_file: str) -> Dict[str, List[int]]:
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)

    if len(structure) > 1:
        logging.warning('Multiple models found in PDB file. Using the first model')
    
    chain_residues = defaultdict(list)
    # Only process the first model
    for model in structure:
        for chain in model:
            for residue in chain:
                if PDB.is_aa(residue):
                    chain_residues[chain.id].append(residue.id[1])
        break  # Exit after processing the first model
    
    return chain_residues

def find_segments(residue_list: List[int], design_ranges: List[List[int]]) -> List[Tuple[int, int, bool]]:
    segments = []
    design_ranges = sorted([range(start, end+1) for start, end in 
                            [(r[0], r[-1]) for r in design_ranges]], key=lambda x: x[0])
    
    for i in range(len(design_ranges) - 1):
        if design_ranges[i][1] >= design_ranges[i+1][0]:
            raise ValueError('Design ranges overlap')

    
    # Group continuous residues
    continuous_segments = []
    for k, g in groupby(enumerate(sorted(residue_list)), lambda ix: ix[0] - ix[1]):
        group = list(map(itemgetter(1), g))
        continuous_segments.append((group[0], group[-1]))

    for start, end in continuous_segments:
        for design_range in design_ranges:
            if (design_range.start >= start and design_range.start <= end and design_range.stop > end) or (design_range.stop >= start and design_range.stop <= end and design_range.start < start):
                raise NotImplementedError(f'Design range between two discontinuous segments: Continious segments are {continuous_segments}, design range is {design_range}')
    
    for start, end in continuous_segments:
        current = start
        for design_range in design_ranges:
            if current <= design_range.start and end > design_range.stop:
                if current < design_range.start:
                    segments.append((current, design_range.start - 1, False))
                    segments.append((design_range.start, design_range.stop, True))
                    current = design_range.stop + 1
                elif current == design_range.start:
                    segments.append((design_range.start, design_range.stop, True))
                    current = design_range.stop + 1
        if current <= end:
            segments.append((current, end, False))
            current = end + 1
    
    return segments


def generate_chain_info(chain_residues: Dict[str, List[int]], design_dict, cdr_length_data: Optional[Dict] = None, design_chain: Optional[Union[str, List[str]]] = None, fixed_chains: Optional[Union[str, List[str]]] = None, use_cdr_range: bool = True) -> List[str]:
    chain_infos = []
    chain_infos_dict = defaultdict(list)
    if isinstance(design_dict, PosixPath):
        with open(design_dict, 'r') as f:
            dict_to_design = json.load(f)
    else:
        dict_to_design = design_dict

    if 'antigen_chain' not in dict_to_design:
        logging.warning("Antigen chain not specified. Assuming all chains except heavy and light chains are antigen chains.")
        dict_to_design['antigen_chain'] = [chain for chain in chain_residues.keys() if chain != dict_to_design['heavy_chain'] and chain != dict_to_design['light_chain']]
    elif isinstance(dict_to_design['antigen_chain'], str):
        dict_to_design['antigen_chain'] = [dict_to_design['antigen_chain']]
    
    # Convert design_chain and fixed_chains to sets for easier checking
    design_chains = set([design_chain] if isinstance(design_chain, str) else design_chain) if design_chain else set([chain for chain in chain_residues.keys() if chain == dict_to_design['heavy_chain'] or chain == dict_to_design['light_chain']])
    fixed_chains = set([fixed_chains] if isinstance(fixed_chains, str) else fixed_chains) if fixed_chains else set([chain for chain in chain_residues.keys() if (chain not in design_chains) and (chain == dict_to_design['heavy_chain'] or chain == dict_to_design['light_chain'] or chain in dict_to_design['antigen_chain'])])
    
    logging.info(f"Designing chains: {design_chains}")
    logging.info(f"Fixing chains: {fixed_chains}")

    # Check for conflicts between design_chains and fixed_chains
    if design_chains.intersection(fixed_chains):
        raise ValueError(f"Chains {design_chains.intersection(fixed_chains)} are specified in both design_chain and fixed_chains.")

    for chain, residues in chain_residues.items():
        if chain == 'H' or chain == dict_to_design['heavy_chain']:
            chain_to_design = 'H'
        elif chain == 'L' or chain == dict_to_design['light_chain']:
            chain_to_design = 'L'
        else:
            chain_to_design = chain
        
        # Skip chains that are neither designed nor fixed, unless we're using all chains
        if chain_to_design not in design_chains and chain_to_design not in fixed_chains:
            continue
        
        if chain_to_design in design_chains:
            design_ranges = [range for key, range in dict_to_design.items() if key.startswith(f"CDR{chain_to_design}")]
        else:
            design_ranges = []  # Empty list for fixed chains
        
        segments = find_segments(residues, design_ranges)
        
        chain_parts = []
        count_design = 0
        for start, end, is_design in segments:
            if is_design and (chain_to_design in design_chains):
                design_length = end - start + 1
                cdr_key = f"CDR{chain_to_design}{count_design + 1}"
                count_design += 1

                if cdr_length_data and cdr_key in cdr_length_data and use_cdr_range:
                    # Option 1: Use CDR length range from data
                    min_length = cdr_length_data[cdr_key]['min_length']
                    max_length = cdr_length_data[cdr_key]['max_length']
                    
                    if design_length < min_length or design_length > max_length:
                        logging.warning(f"Initial length {design_length} for {cdr_key} is outside the specified range [{min_length}, {max_length}]")
                    
                    logging.info(f"Using CDR length range for {cdr_key}: [{min_length}, {max_length}]")
                elif cdr_length_data and use_cdr_range:
                    raise ValueError(f"CDR Length metadata has been provided but does not contain {cdr_key} - check your file!")
                else:
                    # Option 2: Use current length +/- 1
                    min_length = max(1, design_length - 1)  # Ensure min_length is at least 1
                    max_length = design_length + 1
                    logging.info(f"Using current length +/- 1 for {cdr_key}: [{min_length}, {max_length}]")

                chain_parts.append(f"{chain}{start}-{end}/{min_length}-{max_length}")
            else:
                chain_parts.append(f"{chain}{start}-{end}")
        
        chain_infos.append('/'.join([v if '/' not in v else v.split('/')[1] for v in chain_parts]))

        chain_infos_dict['fixed'].extend([v for v in chain_parts if '/' not in v])
        chain_infos_dict['designed'].extend([v for v in chain_parts if '/' in v])
    
    return chain_infos, chain_infos_dict

def generate_contigmap(chain_infos: List[str]) -> str:
    return '[' + '/0 '.join(chain_infos) + '/0 ]'

def generate_command(args, chain_infos: List[str]) -> str:
    contigmap = generate_contigmap(chain_infos)
    command = (
        f'mamba run -n SE3nv '
        f'{os.path.dirname(os.path.realpath(__file__))}/scripts/run_inference.py '
        f'inference.input_pdb={args.input_pdb} '
        f'inference.output_prefix={args.output_prefix} '
        f'"contigmap.contigs={contigmap}" '
        f'inference.num_designs={args.num_designs} '
        f'denoiser.noise_scale_ca={args.noise_scale_ca} '
        f'denoiser.noise_scale_frame={args.noise_scale_frame} '
        # f'diffuser.partial_T={args.partial_T}'
        f'diffuser.T={args.nb_diffusion_steps} '
    )
    if args.cryoem_map:
        command += f'potentials.cryoem_map={args.cryoem_map} '
        command += f'potentials.cryoem_weight={args.cryoem_weight} '
        command += f'potentials.cryoem_aggr={args.cryoem_aggr} '
        command += f'potentials.guiding_potentials=[\"type:densities\"] '
    # else:
    #     command += f'potentials.guiding_potentials=[\"type:interface_ncontacts\"] '
    if args.revert_init_coords:
        command += f'inference.revert_init_coords=True '

    return command
