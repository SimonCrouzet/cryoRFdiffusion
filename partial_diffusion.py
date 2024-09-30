
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


