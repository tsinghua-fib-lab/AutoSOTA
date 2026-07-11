# MIT License
#
# Copyright (c) 2025 tummfm
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""Helper scripts to set up LAMMPS simulations."""

import argparse

import mdtraj

from jax_md_mod import io

__LAMMPS_HEADER = """
# LAMMPS data file written by chemutils
{atoms} atoms
{atomtypes} atom types
0.0 {bx:.4f} xlo xhi
0.0 {by:.4f} ylo yhi
0.0 {bz:.4f} zlo zhi
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", type=str, help="Command to run")
    parser.add_argument("infile", type=str, help="Input file")
    parser.add_argument("outfile", type=str, help="Output file")
    parser.add_argument("--scale_pos", type=float, default=10.0, help="Scale factor for positions")
    args = parser.parse_args()

    if args.command == "write_lmpdat":
        write_lmpdat(args.infile, args.outfile, scale_pos=args.scale_pos)
    else:
        raise ValueError(f"Unknown command: {args.command}")


def write_lmpdat(infile, outfile, scale_pos=10.0):

    box, coords, masses, species = io.load_box(infile)

    # Convert to LAMMPS format
    with open(outfile, "w") as f:
        f.write(__LAMMPS_HEADER.format(
            atoms=coords.shape[0],
            atomtypes=species.max(),
            bx=box[0] * scale_pos,
            by=box[1] * scale_pos,
            bz=box[2] * scale_pos
        ))

        f.write("\nMasses\n\n")
        for i in range(1, species.max() + 1):
            elem = mdtraj.element.Element.getByAtomicNumber(i)
            f.write(f"{i} {elem.mass:.4f}\n")

        f.write("\nAtoms\n\n")
        for i, (coord, spec) in enumerate(zip(coords, species)):
            f.write(f"{i + 1} {spec} {coord[0] * scale_pos:.4f} {coord[1] * scale_pos:.4f} {coord[2] * scale_pos:.4f}\n")

if __name__ == "__main__":
    main()
