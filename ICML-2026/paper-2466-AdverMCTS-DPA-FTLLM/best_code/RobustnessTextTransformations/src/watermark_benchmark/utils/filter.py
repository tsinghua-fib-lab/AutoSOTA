import sys
from dataclasses import replace

from watermark_benchmark.utils.classes import Generation, WatermarkSpec

def run(generations_file, watermarks_file):
    """ Filter generations to only keep those with watemrarks from in the watermark spec """
    generations = Generation.from_file(generations_file)
    model = generations[0].watermark.tokenizer
    with open(watermarks_file) as infile:
        watermarks = {str(replace(WatermarkSpec.from_str(l.strip()), tokenizer=model).to_dict(True, True)) for l in infile.read().split("\n") if len(l)}

    filt_gens = [g for g in generations if g.watermark if not None and str(g.watermark.to_dict(True, True)) in watermarks]

    Generation.to_file("filtered.tsv", filt_gens)


def main():
    if len(sys.argv) != 3:
        return
    run(sys.argv[1], sys.argv[2])


if __name__ == '__main__':
    main()