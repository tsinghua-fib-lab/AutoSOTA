from Bio import PDB
import sys

def extract_chains(input_pdb, output1_pdb, chains1, output2_pdb, chains2):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("input", input_pdb)

    io = PDB.PDBIO()

    # Save chains for first output
    class SelectChains1(PDB.Select):
        def accept_chain(self, chain):
            return chain.id in chains1

    io.set_structure(structure)
    io.save(output1_pdb, select=SelectChains1())

    # Save chains for second output
    class SelectChains2(PDB.Select):
        def accept_chain(self, chain):
            return chain.id in chains2

    io.set_structure(structure)
    io.save(output2_pdb, select=SelectChains2())

def renumber_pdb(input_path, output_path):
    # Maps chain_id -> { original_resseq_str : new_int }
    residue_maps = {}
    # Next new residue ID per chain
    next_id = {}

    with open(input_path) as inp, open(output_path, 'w') as out:
        for line in inp:
            record = line[:6]
            if record in ('ATOM  ', 'HETATM'):
                chain = line[21]
                orig = line[22:26].strip()
                if chain not in residue_maps:
                    residue_maps[chain] = {}
                    next_id[chain] = 1
                if orig not in residue_maps[chain]:
                    residue_maps[chain][orig] = next_id[chain]
                    next_id[chain] += 1
                new = residue_maps[chain][orig]
                # pad to width 4, right-justified
                new_str = str(new).rjust(4)
                line = line[:22] + new_str + line[26:]
                out.write(line)

            elif record == 'TER   ':
                # also renumber the TER record
                chain = line[21]
                orig = line[22:26].strip()
                if chain in residue_maps and orig in residue_maps[chain]:
                    new = residue_maps[chain][orig]
                    new_str = str(new).rjust(4)
                    line = line[:22] + new_str + line[26:]
                out.write(line)

            else:
                # leave all other lines (HEADER, CONECT, END, etc.) unchanged
                out.write(line)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} input.pdb output.pdb")
        sys.exit(1)
    renumber_pdb(sys.argv[1], sys.argv[2])