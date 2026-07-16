import sys
from watermark_benchmark.utils.classes import Generation


def run(filename, spec_files):
    """ Split generations into sub files according to spec files """

    generations = Generation.from_file(filename)
    specs = {}
    cmb = set()
    for s in spec_files:
        with open(s) as infile:
            specs[int(s.split('_')[-1])] = {t.strip() for t in infile.read().split("\n") if len(t)}

    for v in specs.values():
        for vv in v:
            cmb.add(vv)

    split = {k: [] for k in specs}
    for gen in generations:
        key = str(gen.watermark).strip()
        if key not in cmb:
            continue
        for k, sk in specs.items():
            if key in sk:
                split[k].append(gen)
                break


    for k in split:
        Generation.to_file(f"generations_{k}.tsv", split[k])


def main():
    run(sys.argv[1], sys.argv[2:])

if __name__ == '__main__':
    main()
