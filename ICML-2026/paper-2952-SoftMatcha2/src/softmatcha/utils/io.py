import fileinput
import simdjson
from typing import Generator
# from tqdm import tqdm
from softmatcha.utils.custom_tqdm import CustomTqdm

def read_lines(
	inputs: str = "-", jsonl_key: str | None = None
) -> Generator[str, None, None]:
	"""Read and yield lines.

	Args:
		inputs (str): Input file. `-` reads from the standard input.
		jsonl_key (str, optional): If specified, read the text from the key.

	Yields:
		str: A line.
	"""
	if jsonl_key is not None:
		parser = simdjson.Parser()
		with fileinput.input(files=[inputs], mode="rb") as f:
			for line in f:
				yield parser.parse(line).at_pointer(f"/{jsonl_key}")
	else:
		with fileinput.input(
			files=[inputs], mode="r", openhook=fileinput.hook_encoded("utf-8")
		) as f:
			for line in f:
				yield line.rstrip()


def buffer_lines(
	inputs: str = "-",
	buffer_size: int = 1000,
	total: int = 1000,
	chunk: int = 1000,
	jsonl_key: str | None = None,
) -> Generator[list[str], None, None]:
	bar = CustomTqdm(
		total=total,
		bar_format="{bar:64} {n_fmt}/{total_fmt} ETA {remaining}",
		ascii="░█",
		dynamic_ncols=True
	)

	# jsonl_key が存在する場合，まだバグっている
	if jsonl_key is not None:
		parser = simdjson.Parser()
		with fileinput.input(files=[inputs], mode="rb") as f:
			buffer = []
			for line in f:
				buffer.append(parser.parse(line).at_pointer(f"/{jsonl_key}"))
				if len(buffer) >= buffer_size:
					yield buffer
					buffer = []
					bar.update(f.tell() / chunk - bar.n)
			if len(buffer) > 0:
				yield buffer
				bar.update(bar.total - bar.n)

	# jsonl_key が存在しない場合
	else:
		with fileinput.input(files=[inputs], mode="rb") as f:
			buffer = []
			current_batch_bytes = 0
			for line_bytes in f:
				current_batch_bytes += len(line_bytes)
				line_str = line_bytes.decode("utf-8")
				buffer.append(line_str.rstrip())
				if len(buffer) >= buffer_size:
					yield buffer
					buffer = []
					bar.update(current_batch_bytes // chunk - bar.n)
			if len(buffer) > 0:
				yield buffer
				bar.update(bar.total - bar.n)
