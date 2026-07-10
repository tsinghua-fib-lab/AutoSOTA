import logging
import sys
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import simple_parsing
from simple_parsing import choice, field
from simple_parsing.wrappers.field_wrapper import FieldWrapper

from softmatcha import registry
from softmatcha.embeddings import GENSIM_PRETRAINED_MODELS
from softmatcha.tokenizers import get_tokenizer

logger = logging.getLogger(__name__)


class ArgumentParser(simple_parsing.ArgumentParser):
	def preprocess_parser(self) -> None:
		"""Preprocess ArgumentParser."""
		self.parse_known_args_preprocess(sys.argv[1:])

	def parse_known_args_preprocess(
		self,
		args: Sequence[str] | None = None,
		namespace: Namespace | None = None,
		attempt_to_reorder: bool = False,
	) -> None:
		# default Namespace built from parser defaults
		if namespace is None:
			namespace = Namespace()
		if self.config_path:
			if isinstance(self.config_path, Path):
				config_paths = [self.config_path]
			else:
				config_paths = self.config_path
			for config_file in config_paths:
				self.set_defaults(config_file)

		if self.add_config_path_arg:
			temp_parser = ArgumentParser(
				add_config_path_arg=False,
				add_help=False,
				add_option_string_dash_variants=FieldWrapper.add_dash_variants,
				argument_generation_mode=FieldWrapper.argument_generation_mode,
				nested_mode=FieldWrapper.nested_mode,
			)
			temp_parser.add_argument(
				"--config_path",
				type=Path,
				default=self.config_path,
				help="Path to a config file containing default values to use.",
			)
			args_with_config_path, args = temp_parser.parse_known_args(args)
			config_path = args_with_config_path.config_path

			if config_path is not None:
				config_paths = (
					config_path if isinstance(config_path, list) else [config_path]
				)
				for config_file in config_paths:
					self.set_defaults(config_file)

			# Adding it here just so it shows up in the help message. The default will be set in
			# the help string.
			if self._option_string_actions.get("--config_path", None) is None:
				self.add_argument(
					"--config_path",
					type=Path,
					default=config_path,
					help="Path to a config file containing default values to use.",
				)

		assert isinstance(args, list)
		self._preprocessing(args=args, namespace=namespace)

	def parse_known_args(
		self,
		args: Sequence[str] | None = None,
		namespace: Namespace | None = None,
		attempt_to_reorder: bool = False,
	):
		# NOTE: since the usual ArgumentParser.parse_args() calls
		# parse_known_args, we therefore just need to overload the
		# parse_known_args method to support both.
		if args is None:
			# args default to the system args
			args = sys.argv[1:]
		else:
			# make sure that args are mutable
			args = list(args)

		self.parse_known_args_preprocess(
			args=args, namespace=namespace, attempt_to_reorder=attempt_to_reorder
		)
		logger.debug(
			f"Parser {id(self)} is parsing args: {args}, namespace: {namespace}"
		)
		parsed_args, unparsed_args = super(
			simple_parsing.ArgumentParser, self
		).parse_known_args(args, namespace)

		if unparsed_args and self._subparsers and attempt_to_reorder:
			logger.warning(
				f"Unparsed arguments when using subparsers. Will "
				f"attempt to automatically re-order the unparsed arguments "
				f"{unparsed_args}."
			)
			index_in_start = args.index(unparsed_args[0])
			# Simply 'cycle' the args to the right ordering.
			new_start_args = args[index_in_start:] + args[:index_in_start]
			parsed_args, unparsed_args = super(
				simple_parsing.ArgumentParser, self
			).parse_known_args(new_start_args)

		parsed_args = self._postprocessing(parsed_args)
		return parsed_args, unparsed_args


class DataclassWrapperNoDescription(simple_parsing.wrappers.DataclassWrapper):
	@property
	def description(self) -> str:
		return ""


@dataclass
class CommonArguments:
	"""Common arguments."""

	# Backend framework for embeddings.
	backend: str = choice(*registry.get_registry("embedding").keys(), default="gensim")
	# Model name or path for embeddings.
	model: str = field(default="glove-wiki-gigaword-300")


@dataclass
class OutputArguments:
	"""Output arguments."""

	# Output json with metadata instead of matched lines.
	json: bool = False
	# Profile the runtime statistics.
	profile: bool = False
	# Output log to stderr or the specified file.
	log: str | None = field(default=None, nargs="?", const="-")
	# Print line number with output lines.
	line_number: bool = field(default=False, alias=["-n"], action="store_true")
	# Print file name with output lines.
	with_filename: bool = field(default=False, alias=["-H"], action="store_true")
	# Suppress the file name prefix on output.
	no_filename: bool = field(default=False, action="store_true")
	# Output only matched patterns.
	only_matching: bool = field(default=False, alias=["-o"], action="store_true")
	# No output.
	quiet: bool = field(default=False, alias=["-q"], action="store_true")


def get_argparser(args: Sequence[str] | None = None) -> ArgumentParser:
	meta_parser = ArgumentParser(add_help=False)
	meta_parser.add_arguments(CommonArguments, "common")
	known_args, _ = meta_parser.parse_known_args(args=args)
	tokenizer_type = get_tokenizer(known_args.common.backend)

	parser = ArgumentParser(add_help=True)
	parser.epilog = f"""
	Available pre-trained gensim models: {GENSIM_PRETRAINED_MODELS}
	"""

	parser.add_arguments(CommonArguments, "common")

	parser.add_arguments(
		tokenizer_type.Config,
		"tokenizer",
		dataclass_wrapper_class=DataclassWrapperNoDescription,
	)
	return parser