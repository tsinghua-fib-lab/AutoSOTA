import os
import time
import subprocess
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def calc_affinity_crossdocked(sml, name_protein, dir_out='./', is_del=True):
    """
    Return: (docking_score, pose_string)
    """

    SMINA_BIN = '/usr/local/bin/smina'
    FILE_PROTEIN = f'./datasets/crossdocked/structure-files-test/{name_protein}-protein.pdb'
    FILE_LIG_REF = f'./datasets/crossdocked/structure-files-test/{name_protein}-ligand.sdf'


    autobox_add = '1'
    seed = '1234'
    exhaustiveness = '32'


    affinity = 500.0
    pose_str = ""


    timestamp = f"{time.time()}_{os.getpid()}"
    os.makedirs(dir_out, exist_ok=True)
    file_output = os.path.join(dir_out, f"tmp_lig_{timestamp}.sdf")
    smina_cmd_output = os.path.join(dir_out, f"tmp_log_{timestamp}.txt")
    file_out_pose = os.path.join(dir_out, f"tmp_pose_{timestamp}.sdf")  # 必须唯一

    try:
        mol = Chem.MolFromSmiles(sml)
        m2 = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 1
        status = AllChem.EmbedMolecule(m2, params)
        if status == -1:
            return 500, ""

        AllChem.MMFFOptimizeMolecule(m2)

        w = Chem.SDWriter(file_output)
        w.write(m2)
        w.close()


        launch_args = [
            SMINA_BIN,
            '-r', FILE_PROTEIN,
            '-l', file_output,
            '--autobox_ligand', FILE_LIG_REF,
            '--autobox_add', autobox_add,
            '--seed', seed,
            '--exhaustiveness', exhaustiveness,
            '--cpu', '30',
            '-o', file_out_pose,
            '>>', smina_cmd_output,
            '2>/dev/null'
        ]

        launch_string = ' '.join(launch_args)

        p = subprocess.Popen(launch_string, shell=True, stdout=subprocess.PIPE)
        p.communicate(timeout=1800)


        found_score = False
        if os.path.exists(smina_cmd_output):
            with open(smina_cmd_output, 'r') as f:
                for lines in f.readlines():
                    lines = lines.split()
                    if len(lines) == 4 and lines[0] == '1':
                        affinity = float(lines[1])
                        found_score = True


        if found_score and os.path.exists(file_out_pose):
            with open(file_out_pose, 'r') as f:
                pose_str = f.read()
        else:
            affinity = 500.0
            pose_str = ""

    except Exception as e:
        print(f"Error: {e}")
        affinity = 500.0
        pose_str = ""

    finally:
        if is_del:
            for f in [file_output, smina_cmd_output, file_out_pose]:
                if os.path.exists(f):
                    os.remove(f)

    return affinity, pose_str