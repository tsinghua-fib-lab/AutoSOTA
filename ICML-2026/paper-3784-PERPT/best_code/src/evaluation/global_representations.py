#!/usr/bin/env python3
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import pyBigWig
from pyfaidx import Fasta
import json
from tqdm import tqdm
import pandas as pd
import logging
import sys
import random
import seaborn as sns
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
sys.path.append("../gamba")
from torch.nn import MSELoss, CrossEntropyLoss
from sequence_models.constants import MSA_PAD, START, STOP
from evodiff.utils import Tokenizer
from gamba.constants import TaskType, DNA_ALPHABET_PLUS
from gamba.collators import gLMCollator, gLMMLMCollator
from gamba.model import create_model, JambagambaModel, JambaGambaNoConsModel, JambaGambaNOALMModel
from my_caduceus.configuration_caduceus import CaduceusConfig
from my_caduceus.modeling_caduceus import (
    CaduceusConservationForMaskedLM,
    CaduceusForMaskedLM,
    CaduceusConservation
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class GTFParser:
    """Parser for GTF annotation files."""
    
    def __init__(self, gtf_files):
        """
        Initialize GTF parser with one or more GTF files.
        
        Args:
            gtf_files: List of paths to GTF files
        """
        self.features = {
            'coding_regions': [],
            'noncoding_regions': [],
            'exons': [],
            'introns': [],
            'promoters': []
        }
        
        # Parse GTF files
        self._parse_gtf_files(gtf_files)
    
    def _parse_attributes(self, attr_string):
        """Parse GTF attribute string into a dictionary."""
        attrs = {}
        for attr in attr_string.strip().split(';'):
            if not attr.strip():
                continue
            key, value = attr.strip().split(' ', 1)
            attrs[key] = value.strip('"')
        return attrs
    
    def _parse_gtf_files(self, gtf_files):
        """Parse GTF files and extract features."""
        logging.info(f"Parsing {len(gtf_files)} GTF files...")
        
        for gtf_file in gtf_files:
            logging.info(f"Parsing GTF file: {gtf_file}")
            
            # Dictionary to temporarily store genes and transcripts
            genes = {}
            transcripts = {}
            
            with open(gtf_file, 'r') as f:
                for line in tqdm(f, desc=f"Reading {os.path.basename(gtf_file)}"):
                    if line.startswith('#'):
                        continue
                    
                    fields = line.strip().split('\t')
                    if len(fields) < 9:
                        continue
                    
                    chrom, source, feature_type, start, end, score, strand, frame, attributes = fields
                    start, end = int(start), int(end)
                    
                    try:
                        attrs = self._parse_attributes(attributes)
                    except:
                        # Skip lines with malformed attributes
                        continue
                    
                    # Process different feature types
                    if feature_type == 'gene':
                        gene_id = attrs.get('gene_id')
                        gene_type = attrs.get('gene_type')
                        
                        if gene_id and gene_type:
                            genes[gene_id] = {
                                'chrom': chrom,
                                'start': start,
                                'end': end,
                                'strand': strand,
                                'type': gene_type,
                                'transcripts': []
                            }
                    
                    elif feature_type == 'transcript':
                        gene_id = attrs.get('gene_id')
                        transcript_id = attrs.get('transcript_id')
                        transcript_type = attrs.get('transcript_type')
                        
                        if gene_id and transcript_id and transcript_type:
                            is_canonical = 'tag "Ensembl_canonical"' in attributes or 'tag "CCDS"' in attributes
                            
                            transcripts[transcript_id] = {
                                'gene_id': gene_id,
                                'chrom': chrom,
                                'start': start,
                                'end': end,
                                'strand': strand,
                                'type': transcript_type,
                                'is_canonical': is_canonical,
                                'exons': []
                            }
                            
                            if gene_id in genes:
                                genes[gene_id]['transcripts'].append(transcript_id)
                    
                    elif feature_type == 'exon':
                        transcript_id = attrs.get('transcript_id')
                        
                        if transcript_id and transcript_id in transcripts:
                            transcripts[transcript_id]['exons'].append({
                                'start': start,
                                'end': end
                            })
                    
                    elif feature_type == 'CDS':
                        transcript_id = attrs.get('transcript_id')
                        
                        if transcript_id and transcript_id in transcripts:
                            if 'cds_regions' not in transcripts[transcript_id]:
                                transcripts[transcript_id]['cds_regions'] = []
                            
                            transcripts[transcript_id]['cds_regions'].append({
                                'start': start,
                                'end': end
                            })
            
            # Process and organize the parsed data
            for transcript_id, transcript in transcripts.items():
                if not transcript['is_canonical']:
                    continue
                
                chrom = transcript['chrom']
                
                # Add coding regions (CDS)
                if 'cds_regions' in transcript and transcript['type'] == 'protein_coding':
                    for cds in transcript['cds_regions']:
                        self.features['coding_regions'].append({
                            'chrom': chrom,
                            'start': cds['start'],
                            'end': cds['end'],
                            'strand': transcript['strand'],
                            'gene_id': transcript['gene_id'],
                            'transcript_id': transcript_id
                        })
                
                # Add exons
                for exon in transcript['exons']:
                    self.features['exons'].append({
                        'chrom': chrom,
                        'start': exon['start'],
                        'end': exon['end'],
                        'strand': transcript['strand'],
                        'gene_id': transcript['gene_id'],
                        'transcript_id': transcript_id
                    })
                
                # Add non-coding regions (exons without CDS in protein-coding genes or 
                # all exons in non-coding genes)
                if transcript['type'] != 'protein_coding':
                    for exon in transcript['exons']:
                        self.features['noncoding_regions'].append({
                            'chrom': chrom,
                            'start': exon['start'],
                            'end': exon['end'],
                            'strand': transcript['strand'],
                            'gene_id': transcript['gene_id'],
                            'transcript_id': transcript_id
                        })
                
                # Derive introns from exons
                if len(transcript['exons']) >= 2:
                    sorted_exons = sorted(transcript['exons'], key=lambda x: x['start'])
                    for i in range(len(sorted_exons) - 1):
                        intron_start = sorted_exons[i]['end'] + 1
                        intron_end = sorted_exons[i + 1]['start'] - 1
                        
                        if intron_end > intron_start:
                            self.features['introns'].append({
                                'chrom': chrom,
                                'start': intron_start,
                                'end': intron_end,
                                'strand': transcript['strand'],
                                'gene_id': transcript['gene_id'],
                                'transcript_id': transcript_id
                            })
                
                # Add promoter regions (2kb upstream of TSS)
                if transcript['strand'] == '+':
                    promoter_start = max(1, transcript['start'] - 2000)
                    promoter_end = transcript['start'] - 1
                else:
                    promoter_start = transcript['end'] + 1
                    promoter_end = transcript['end'] + 2000
                
                if promoter_end > promoter_start:
                    self.features['promoters'].append({
                        'chrom': chrom,
                        'start': promoter_start,
                        'end': promoter_end,
                        'strand': transcript['strand'],
                        'gene_id': transcript['gene_id'],
                        'transcript_id': transcript_id
                    })
        
        # Log summary of parsed features
        for feature_type, features in self.features.items():
            logging.info(f"Found {len(features)} {feature_type}")
    
    def get_regions_by_type(self, feature_type, chrom=None, min_length=None, max_length=None):
        """
        Get genomic regions of a specific feature type.
        
        Args:
            feature_type: Type of feature (e.g., 'coding_regions', 'exons')
            chrom: Optional chromosome filter
            min_length: Minimum region length
            max_length: Maximum region length
            
        Returns:
            List of regions matching criteria
        """
        if feature_type not in self.features:
            return []
        
        filtered_regions = []
        for region in self.features[feature_type]:
            if chrom and region['chrom'] != chrom:
                continue
                
            length = region['end'] - region['start'] + 1
            
            if min_length and length < min_length:
                continue
                
            if max_length and length > max_length:
                continue
                
            filtered_regions.append(region)
            
        return filtered_regions


def get_latest_dcp_checkpoint_path(ckpt_dir, last_step=-1):
    """Find the latest checkpoint path."""
    ckpt_path = None
    if last_step == -1:
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir, exist_ok=True)
        for dir_name in os.listdir(ckpt_dir):
            if "dcp_" in dir_name:
                step = int(dir_name.split("dcp_")[-1])
                if step > last_step:
                    ckpt_path = os.path.join(ckpt_dir, dir_name)
                    last_step = step
    else:
        ckpt_path = os.path.join(ckpt_dir, f"dcp_{last_step}")
    return ckpt_path


def load_model(
    checkpoint_dir,
    config_fpath,
    last_step=44000,
    model_type="gamba",
    training_task="dual",
    device=None
):
    """Load Gamba or Caduceus model depending on model_type and training_task."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = Tokenizer(DNA_ALPHABET_PLUS)

    if model_type == "gamba":
        # Resolve checkpoint path
        if training_task == "dual":
            ckpt_path = os.path.join(checkpoint_dir, f"dcp_{last_step}")
        elif training_task == "seq_only":
            ckpt_path = os.path.join(checkpoint_dir, f"dcp_nocons_{last_step}")
        else:
            ckpt_path = get_latest_dcp_checkpoint_path(checkpoint_dir, f"noALM{last_step}")

        # Load config
        with open(config_fpath, "r") as f:
            config = json.load(f)

        task = TaskType(config["task"].lower().strip())

        logging.info(f"Gamba Model | Task: {task}, Type: {config['model_type']}")

        # Create base model
        base_model, _ = create_model(
            task,
            config["model_type"],
            config["model_config"],
            tokenizer.mask_id.item(),
        )

        # Set up Gamba model wrapper
        d_model = config.get("d_model", 512)
        nhead = config.get("n_head", 8)
        n_layers = config.get("n_layers", 6)
        dim_feedforward = config.get("dim_feedforward", d_model)
        padding_id = config.get("padding_id", 0)

        if training_task == "dual":
            model = JambagambaModel(
                base_model,
                d_model=d_model,
                nhead=nhead,
                n_layers=n_layers,
                padding_id=padding_id,
                dim_feedfoward=dim_feedforward,
            )
        elif training_task == "seq_only":
            model = JambaGambaNoConsModel(
                base_model,
                d_model=d_model,
                nhead=nhead,
                n_layers=n_layers,
                padding_id=padding_id,
                dim_feedfoward=dim_feedforward,
            )
        else:
            model = JambaGambaNOALMModel(
                base_model,
                d_model=d_model,
                nhead=nhead,
                n_layers=n_layers,
                padding_id=padding_id,
                dim_feedfoward=dim_feedforward,
            )

        # Load checkpoint
        logging.info(f"Loading Gamba trained on {training_task} from {ckpt_path}")
        checkpoint = torch.load(os.path.join(ckpt_path, "model_optimizer.pt"), map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        model.to(device)
        model.eval()
        return model, tokenizer

    elif model_type == "caduceus":

        config = CaduceusConfig(
            d_model=256,
            n_layer=8,
            vocab_size=len(DNA_ALPHABET_PLUS)
        )

        if training_task == "dual":
            model = CaduceusConservationForMaskedLM(config)
            ckpt_path = os.path.join(checkpoint_dir, f"dcp_conscaduceus_{last_step}")
        elif training_task == "seq_only":
            model = CaduceusForMaskedLM(config)
            ckpt_path = os.path.join(checkpoint_dir, f"dcp_{last_step}")
        else:
            model = CaduceusConservation(config)
            ckpt_path = get_latest_dcp_checkpoint_path(checkpoint_dir, f"consONLYcaduceus_60000")

        logging.info(f"Loading Caduceus checkpoint from {ckpt_path}")
        checkpoint = torch.load(os.path.join(ckpt_path, "model_optimizer.pt"), map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        model.to(device)
        model.eval()
        return model, tokenizer

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


def get_bigwig_values(bw, chrom, start, end):
    """
    Get values from bigWig file using intervals method.
    
    Args:
        bw: pyBigWig object
        chrom: Chromosome name
        start: Start position (0-based)
        end: End position (exclusive)
        
    Returns:
        numpy array of values
    """
    # Initialize vals with zeros
    vals = np.zeros(end - start, dtype=np.float64)
    
    try:
        # Get intervals from the bigwig file
        intervals = bw.intervals(chrom, start, end)
        
        # Check if intervals is None
        if intervals is None:
            # Return zeros if no intervals found
            return vals
        
        # Fill in values from intervals
        for interval_start, interval_end, value in intervals:
            rel_start = max(0, interval_start - start)
            rel_end = min(end - start, interval_end - start)
            vals[rel_start:rel_end] = value
            
        return vals
    except Exception as e:
        logging.debug(f"Error getting values for {chrom}:{start}-{end}: {e}")
        return vals
    

def get_phylop_score_ranges(bigwig_file, chromosomes, num_samples=1000, region_length=50):
    """
    Get ranges of phyloP scores by sampling random regions.
    
    Args:
        bigwig_file: Path to bigwig file with phyloP scores
        chromosomes: List of chromosomes to sample from
        num_samples: Number of regions to sample
        region_length: Length of each region
    
    Returns:
        Dictionary with information about phyloP score distribution
    """
    logging.info(f"Analyzing phyloP score distribution from {num_samples} random samples...")
    
    bw = pyBigWig.open(bigwig_file)
    
    # Verify the chromosomes exist in the bigwig file
    valid_chroms = []
    for chrom in chromosomes:
        if chrom in bw.chroms():
            valid_chroms.append(chrom)
        else:
            logging.warning(f"Chromosome {chrom} not found in bigwig file")
    
    if not valid_chroms:
        logging.error("No valid chromosomes found in bigwig file")
        raise ValueError("No valid chromosomes found in bigwig file")
    
    all_scores = []
    
    for _ in tqdm(range(num_samples), desc="Sampling phyloP scores"):
        chrom = random.choice(valid_chroms)
        try:
            chrom_length = bw.chroms()[chrom]
            if chrom_length <= region_length:
                continue
            
            # Define start and end positions
            start = random.randint(0, chrom_length - region_length)
            end = start + region_length
            
            # Initialize vals with zeros
            vals = np.zeros(end - start, dtype=np.float64)

            # Get the conservation scores from the bigwig file
            intervals = bw.intervals(chrom, start, end)

            # Check if intervals is None
            if intervals is None:
                print("Error: intervals is None")
                # skip this region
                continue
            else:
                for interval_start, interval_end, value in intervals:
                    vals[interval_start - start : interval_end - start] = value
                    # Get to 2 decimal places
                    vals = np.round(vals, 2)
            
            # Filter valid scores
            valid_scores = vals[~np.isnan(vals)]
            if len(valid_scores) > 0:
                all_scores.extend(valid_scores)
        except Exception as e:
            logging.debug(f"Error sampling scores from {chrom}: {e}")
            continue
    
    # Check if we have any valid scores
    if not all_scores:
        logging.error("No valid scores found during sampling")
        raise ValueError("No valid phyloP scores could be sampled from the provided chromosomes")
    
    all_scores = np.array(all_scores)
    
    # Calculate percentiles
    percentiles = {
        'min': np.min(all_scores),
        'p1': np.percentile(all_scores, 1),
        'p5': np.percentile(all_scores, 5),
        'p25': np.percentile(all_scores, 25),
        'median': np.median(all_scores),
        'p75': np.percentile(all_scores, 75),
        'p95': np.percentile(all_scores, 95),
        'p99': np.percentile(all_scores, 99),
        'max': np.max(all_scores)
    }
    
    # Add p45 and p55 for defining "neutral" range
    percentiles['p45'] = np.percentile(all_scores, 45)
    percentiles['p55'] = np.percentile(all_scores, 55)
    
    bw.close()
    
    logging.info(f"PhyloP score distribution: {percentiles}")
    
    return {
        'negative': (percentiles['min'], percentiles['p5']),
        'neutral': (percentiles['p45'], percentiles['p55']),
        'positive': (percentiles['p95'], percentiles['max']),
        'all_scores': all_scores,
        'percentiles': percentiles
    }

def sample_regions_by_phylop(bigwig_file, genome_fasta, score_range, num_regions=100, max_length=2048, region_length=None, feature_length=1000, chromosomes=None, model_type="gamba"):
    """
    Sample regions with phyloP scores in a specified range.
    Evaluates only the last feature_length bases.
    
    Args:
        bigwig_file: Path to bigwig file with phyloP scores
        genome_fasta: Path to genome fasta file
        score_range: Tuple of (min_score, max_score)
        num_regions: Number of regions to sample
        max_length: Maximum length of each region
        region_length: Legacy parameter for backward compatibility
        feature_length: Length of the "feature" portion at the end to evaluate
        chromosomes: List of chromosomes to sample from
    
    Returns:
        List of sampled regions
    """
    # For backward compatibility
    if region_length is not None:
        max_length = region_length
        
    min_score, max_score = score_range
    logging.info(f"Sampling {num_regions} regions with phyloP scores between {min_score:.4f} and {max_score:.4f}...")
    
    # Open genome and bigwig files
    genome = Fasta(genome_fasta)
    bw = pyBigWig.open(bigwig_file)
    
    # Check if chromosomes exist in both genome and bigwig
    valid_chroms = []
    for chrom in (chromosomes or list(genome.keys())):
        if chrom in genome.keys() and chrom in bw.chroms():
            valid_chroms.append(chrom)
    
    if not valid_chroms:
        logging.error("No valid chromosomes found in both genome and bigwig")
        bw.close()
        return []
    
    # Debug information
    logging.info(f"Using valid chromosomes: {valid_chroms}")
    
    sampled_regions = []
    max_attempts = num_regions * 100  # Limit the number of attempts
    attempts = 0
    
    with tqdm(total=num_regions, desc=f"PhyloP {min_score:.2f}-{max_score:.2f}") as pbar:
        while len(sampled_regions) < num_regions and attempts < max_attempts:
            attempts += 1
            
            # Choose a random chromosome
            chrom = random.choice(valid_chroms)
            chrom_length = len(genome[chrom])
            
            if chrom_length <= max_length:
                continue
            
            # Choose a random start position
            start = random.randint(0, chrom_length - max_length)
            end = start + max_length

            
            try:
                # Get the sequence
                sequence = genome[chrom][start:end].seq
                
                # Get scores for the region using our custom function
                scores = get_bigwig_values(bw, chrom, start, end)
                
                if model_type=="gamba":
                    # Evaluate only the last feature_length portion
                    feature_start_in_window = max(0, len(scores) - feature_length)
                    feature_end_in_window = len(scores)
                else:
                    # Caduceus model, so we're only evaluating the middle
                    feature_start_in_window = max(0, (len(scores) - feature_length) // 2)
                    feature_end_in_window = feature_start_in_window + feature_length
                    
                # Get feature portion scores for selection criteria
                feature_scores = scores[feature_start_in_window:feature_end_in_window]
                
                # Filter non-zero values for mean calculation
                non_zero_scores = feature_scores[feature_scores != 0]
                
                if len(non_zero_scores) > 0:
                    mean_score = np.mean(non_zero_scores)
                    
                    if min_score <= mean_score <= max_score:
                        # Add to sampled regions
                        sampled_regions.append({
                            'chrom': chrom,
                            'start': start,
                            'end': end,
                            'sequence': sequence,
                            'scores': scores,
                            'mean_score': mean_score,
                            'feature_start_in_window': feature_start_in_window,
                            'feature_end_in_window': feature_end_in_window
                        })
                        
                        pbar.update(1)
            except Exception as e:
                logging.debug(f"Error processing {chrom}:{start}-{end}: {e}")
    
    bw.close()
    
    if len(sampled_regions) < num_regions:
        logging.warning(f"Could only sample {len(sampled_regions)} regions with scores between {min_score:.4f} and {max_score:.4f}")
    
    return sampled_regions

def sample_regions_by_feature(bigwig_file, genome_fasta, gtf_parser, feature_type, num_regions=100, max_length=2048, chromosomes=None, model_type="gamba"):
    """
    Sample regions based on genomic features with upstream context.
    
    Args:
        bigwig_file: Path to bigwig file with phyloP scores
        genome_fasta: Path to genome fasta file
        gtf_parser: GTFParser instance
        feature_type: Type of feature to sample from
        num_regions: Number of regions to sample
        max_length: Maximum length of each region (context + feature)
        chromosomes: List of chromosomes to sample from
    
    Returns:
        List of sampled regions
    """
    logging.info(f"Sampling {num_regions} regions from {feature_type}...")
    
    # Open genome and bigwig files
    genome = Fasta(genome_fasta)
    bw = pyBigWig.open(bigwig_file)
    
    # Check for valid chromosomes in both genome and bigwig
    valid_chroms = []
    for chrom in (chromosomes or list(genome.keys())):
        if chrom in genome.keys() and chrom in bw.chroms():
            valid_chroms.append(chrom)
    
    if not valid_chroms:
        logging.error("No valid chromosomes found in both genome and bigwig")
        bw.close()
        return []
    
    # Get features for specified chromosomes
    available_features = []
    for chrom in valid_chroms:
        features = gtf_parser.get_regions_by_type(feature_type, chrom=chrom)
        available_features.extend(features)
    
    logging.info(f"Found {len(available_features)} {feature_type} features in the specified chromosomes")
    
    if not available_features:
        logging.warning(f"No {feature_type} features found for the specified chromosomes")
        bw.close()
        return []
    
    sampled_regions = []
    
    # Try to sample regions
    random.shuffle(available_features)
    
    with tqdm(total=min(num_regions, len(available_features)), desc=f"Feature {feature_type}") as pbar:
        for feature in available_features:
            if len(sampled_regions) >= num_regions:
                break
            
            chrom = feature['chrom']
            feature_start = feature['start']
            feature_end = feature['end']
            feature_length = feature_end - feature_start + 1
            
            if model_type =="gamba":
                # Calculate maximum context we can add
                max_context_length = min(max_length - feature_length, 1000)
                # If feature is too long, just take the last 1000bp
                if feature_length > max_length:
                    if feature['strand'] == '+':
                        start = feature_end - 1000 + 1
                        end = feature_end
                    else:
                        continue  # or handle '-' strand explicitly if needed

                    feature_start_in_window = max_length - 1000
                    feature_end_in_window = max_length

                else:
                    # Add context before the feature
                    if feature['strand'] == '+':
                        context_start = max(1, feature_start - max_context_length)
                        start = context_start
                        end = feature_end
                    else:
                        continue
                    
                    # Calculate feature positions within the window
                    feature_start_in_window = max(0, feature_start - start)
                    feature_end_in_window = max(0, feature_end - start)

            else: 
                # model_type is caduceus – center the feature with context
                chrom_length = len(genome[chrom])
                
                # Clip feature to 1000 bp if too long
                if feature_length > 1000:
                    center = (feature_start + feature_end) // 2
                    feature_length = 1000
                    feature_start = max(0, center - 500)
                    feature_end = feature_start + 1000 - 1

                # Try to give full context, but trim if near chromosome edge
                context_bp = min((max_length - feature_length) // 2, 1000)
                
                start = max(0, feature_start - context_bp)
                end = min(chrom_length, feature_end + context_bp + 1)
                
                # Calculate feature position in the window
                feature_start_in_window = feature_start - start
                feature_end_in_window = feature_end - start 

            try:
                # Get the sequence and scores
                sequence = genome[chrom][start:end].seq
                scores = get_bigwig_values(bw, chrom, start, end)
                
                # Ensure we got valid data
                if len(sequence) >= feature_end_in_window - feature_start_in_window:
                    # Store feature annotations for later evaluation
                    feature_region = {
                        'chrom': chrom,
                        'start': start,
                        'end': end,
                        'sequence': sequence,
                        'scores': scores,
                        'feature_id': feature.get('transcript_id', feature.get('gene_id', 'unknown')),
                        'feature_start_in_window': feature_start_in_window,
                        'feature_end_in_window': feature_end_in_window,
                        'strand': feature['strand']
                    }
                    
                    # Calculate mean score for the feature portion only
                    feature_scores = scores[feature_start_in_window:feature_end_in_window]
                    non_zero_scores = feature_scores[feature_scores != 0]
                    mean_score = np.mean(non_zero_scores) if len(non_zero_scores) > 0 else 0
                    feature_region['mean_score'] = mean_score
                    
                    sampled_regions.append(feature_region)
                    pbar.update(1)
            except Exception as e:
                logging.debug(f"Error processing {chrom}:{start}-{end}: {e}")
    
    bw.close()
    
    if len(sampled_regions) < num_regions:
        logging.warning(f"Could only sample {len(sampled_regions)} regions from {feature_type}")
    
    return sampled_regions

def predict_scores_batched(model, tokenizer, regions, batch_size=8, device=None, model_type="gamba", training_task="dual"):
    """Extract full hidden state representations for each region without masking."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_hidden_states = []
    region_info = []

    logging.info(f"Extracting representations for {len(regions)} regions with batch size {batch_size}...")

    if model_type == "gamba":
        collator = gLMCollator(tokenizer=tokenizer, test=True)
    else:
        collator = gLMMLMCollator(tokenizer=tokenizer, test=True)

    for i in tqdm(range(0, len(regions), batch_size), desc="Batch encoding"):
        batch_regions = regions[i:i + batch_size]
        batch_inputs = []
        batch_region_info = []

        for region in batch_regions:
            sequence_tokens = tokenizer.tokenizeMSA(region['sequence'])
            scores = region['scores']
            fs = region.get('feature_start_in_window', 0)
            fe = region.get('feature_end_in_window', len(scores))

            batch_inputs.append((sequence_tokens, scores))
            batch_region_info.append({
                'chrom': region['chrom'],
                'start': region['start'],
                'end': region['end'],
                'feature_id': region.get('feature_id', 'unknown'),
                'mean_score': region.get('mean_score', 0.0),
                'feature_start_in_window': fs,
                'feature_end_in_window': fe
            })
            region_info.append(batch_region_info[-1])

        if not batch_inputs:
            continue

        if model_type == "gamba":
            collated = collator(batch_inputs)
            with torch.no_grad():
                outputs = model(collated[0].to(device), collated[1].to(device))

            if "representation" in outputs:
                hs = outputs["representation"]  # (B, T, D)
                all_hidden_states.extend(hs.cpu().numpy())  # Directly append the hidden states
            else:
                for _ in range(len(batch_inputs)):
                    all_hidden_states.append(np.full((collated[0].shape[1], model.config.d_model), np.nan))

        elif model_type == "caduceus":
            feature_spans = [(r["feature_start_in_window"], r["feature_end_in_window"]) for r in batch_region_info]
            batch = collator(batch_inputs, region=feature_spans)
            with torch.no_grad():
                sequence_input = batch[0][:, 0, :].long()  # (B, T)
                model_kwargs = {
                    "input_ids": sequence_input.to(device),
                    "output_hidden_states": True
                }
                outputs = model(**model_kwargs)

            if "hidden_states" in outputs:
                hs_all = outputs["hidden_states"]  # list of (B, T, D)
                hs = hs_all[-1]  # final layer (B, T, D)

                for j in range(hs.shape[0]):
                    all_hidden_states.append(hs[j].cpu().numpy())

            else:
                for _ in range(len(batch_inputs)):
                    all_hidden_states.append(np.full((sequence_input.shape[1], model.config.d_model), np.nan))

    return all_hidden_states, region_info

import numpy as np

def extract_region_embeddings(representations, region_info, mode="roi", model_type=None):
    embeddings = []
    labels = []

    for rep, info in zip(representations, region_info):
        if isinstance(rep, float) or rep is None or np.isnan(rep).any():
            continue

        if mode == "roi":
            fs = info["feature_start_in_window"]
            fe = info["feature_end_in_window"]

            if model_type in ["nt", "nt-ms"]:
                fs //= 6
                fe //= 6

            # Defensive bounds check
            if fe > rep.shape[0] or fs < 0 or fe <= fs:
                print(f"[SKIP] ROI bounds invalid: fs={fs}, fe={fe}, rep.shape[0]={rep.shape[0]}")
                continue

            rep_slice = rep[fs:fe]
        elif mode == "full":
            rep_slice = rep
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        if rep_slice.ndim != 2:
            print(f"[SKIP] Invalid rep_slice shape: {rep_slice.shape}")
            continue

        pooled = rep_slice.mean(axis=0)

        if pooled.ndim != 1:
            print(f"[SKIP] Invalid pooled shape: {pooled.shape}")
            continue

        embeddings.append(pooled)
        labels.append(info.get("category", "unknown"))

    # Defensive stack
    try:
        return np.stack(embeddings), labels
    except ValueError as e:
        print(f"[ERROR] np.stack failed: {e}")
        for i, emb in enumerate(embeddings):
            print(f"embedding[{i}] shape: {emb.shape}")
        raise

import umap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_umap(embeddings, labels, output_path, title="UMAP of Representations"):
    umap_model = umap.UMAP()
    embedding_2d = umap_model.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=embedding_2d[:, 0], y=embedding_2d[:, 1], hue=labels, palette="tab10", s=40, alpha=0.8)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import NearestNeighbors

def leave_one_out_1nn_accuracy(embeddings, labels):
    labels = np.array(labels)
    embeddings = np.array(embeddings)

    nn = NearestNeighbors(n_neighbors=2, metric='euclidean').fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)

    # Use the second closest point (excluding the point itself)
    nearest_neighbor_indices = indices[:, 1]
    preds = labels[nearest_neighbor_indices]

    accuracy = np.mean(preds == labels)
    conf_mat = confusion_matrix(labels, preds, labels=np.unique(labels))
    return accuracy, conf_mat

def plot_knn_heatmap(embeddings, labels, output_path, title="1-NN Classification Accuracy"):
    accuracy, conf_mat = leave_one_out_1nn_accuracy(embeddings, labels)
    label_set = np.unique(labels)

    # Normalize confusion matrix rows
    acc_matrix = conf_mat.astype(float) / conf_mat.sum(axis=1, keepdims=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(acc_matrix, xticklabels=label_set, yticklabels=label_set, vmin=0, vmax=0.85,
                cmap="Blues", annot=True, fmt=".2f", cbar_kws={"label": "1-NN Accuracy"})
    plt.title(f"{title} (Acc: {accuracy:.2%})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def analyze_agreement(
    genome_fasta,
    bigwig_file,
    gtf_files,
    checkpoint_dir,
    config_fpath,
    output_dir,
    num_regions=100,
    region_length=2048,
    chromosomes=None,
    last_step=44000,
    batch_size=8,
    training_chromosomes=None,
    test_chromosomes=None,
    training_task = 'dual',
    model_type= 'gamba'
):
    """
    Analyze agreement between predicted and true phyloP scores across different
    genomic regions and chromosomes.
    
    Args:
        genome_fasta: Path to genome FASTA file
        bigwig_file: Path to phyloP bigWig file
        gtf_files: List of paths to GTF files
        checkpoint_dir: Directory containing model checkpoints
        config_fpath: Path to model config JSON
        output_dir: Directory to save outputs
        num_regions: Number of regions to sample per category
        region_length: Length of each region
        chromosomes: List of all chromosomes to analyze
        last_step: Checkpoint step to use
        batch_size: Batch size for model predictions
        training_chromosomes: List of chromosomes used in training
        test_chromosomes: List of chromosomes held out for testing
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Load model
    model, tokenizer = load_model(checkpoint_dir, config_fpath, last_step=last_step, device=device, training_task = training_task, model_type=model_type)
    
    # Parse GTF files
    gtf_parser = GTFParser(gtf_files)
    
    # Get phyloP score distribution
    score_dist = get_phylop_score_ranges(bigwig_file, chromosomes, num_samples=1000)
    
    # Define sampling categories
    categories = [
        # Score-based categories
        #{"name": "negative_phylop", "type": "score", "range": score_dist['negative']},
        {"name": "neutral_phylop", "type": "score", "range": score_dist['neutral']},
        {"name": "positive_phylop", "type": "score", "range": score_dist['positive']},
        
        # Feature-based categories
        {"name": "coding_regions", "type": "feature", "feature": "coding_regions"},
        {"name": "noncoding_regions", "type": "feature", "feature": "noncoding_regions"},
        {"name": "exons", "type": "feature", "feature": "exons"},
        {"name": "introns", "type": "feature", "feature": "introns"},
        {"name": "promoters", "type": "feature", "feature": "promoters"}
    ]

    all_group_embeddings = {
        "training": {"roi": [], "full": [], "labels": []},
        "test": {"roi": [], "full": [], "labels": []}
    }

    
    # Results storage
    all_correlation_results = []
    
    # Set up separate analysis for held out vs training chromosomes
    chromosome_groups = {}
    if training_chromosomes and test_chromosomes:
        chromosome_groups = {
            "training": training_chromosomes,
            "test": test_chromosomes
        }
    else:
        chromosome_groups = {"all": chromosomes}
    
    # Analyze each chromosome group separately
    all_full_embeddings = []
    all_roi_embeddings = []
    all_labels = []
    for group_name, group_chroms in chromosome_groups.items():
        logging.info(f"Analyzing {group_name} chromosomes: {group_chroms}")
        
        # Sample and analyze regions for each category in this chromosome group
        for category in categories:
            logging.info(f"Processing category: {category['name']} for {group_name} chromosomes")
            
            # Sample regions based on category type
            if category['type'] == 'score':
                regions = sample_regions_by_phylop(
                    bigwig_file, genome_fasta, category['range'], 
                    num_regions=num_regions, max_length=region_length, 
                    chromosomes=group_chroms, model_type = model_type
                )
            else:  # feature-based
                regions = sample_regions_by_feature(
                    bigwig_file, genome_fasta, gtf_parser, category['feature'],
                    num_regions=num_regions, max_length=region_length,
                    chromosomes=group_chroms, model_type = model_type
                )
            
            if not regions:
                logging.warning(f"No regions sampled for {category['name']} in {group_name} chromosomes")
                continue
            
            # Get representations for sampled regions
            sequence_representations, region_info = predict_scores_batched(
                model, tokenizer, regions, batch_size=batch_size, device=device, model_type = model_type,
                training_task=training_task
            )
            
            # Tag each region with the current category
            for r in region_info:
                r["category"] = category["name"]

            # Extract embeddings
            full_embeds, full_labels = extract_region_embeddings(sequence_representations, region_info, mode="full")
            roi_embeds, roi_labels = extract_region_embeddings(sequence_representations, region_info, mode="roi")

            # Accumulate for GROUP-level global UMAP/knn
            all_group_embeddings[group_name]["roi"].extend(roi_embeds)
            all_group_embeddings[group_name]["full"].extend(full_embeds)
            all_group_embeddings[group_name]["labels"].extend(full_labels)  # Same as roi_labels
    for group_name, group_data in all_group_embeddings.items():
        print(f"Generating GLOBAL UMAP and 1-NN plots for group: {group_name}")

        # UMAP and heatmap: ROI
        plot_umap(
            group_data["roi"],
            group_data["labels"],
            output_dir / f"global_umap_roi_{group_name}.png",
            title=f"Global UMAP - ROI ({group_name})"
        )

        plot_knn_heatmap(
            group_data["roi"],
            group_data["labels"],
            output_dir / f"global_knn_roi_{group_name}.png",
            title=f"1-NN Accuracy - ROI ({group_name})"
        )

        # UMAP and heatmap: FULL
        plot_umap(
            group_data["full"],
            group_data["labels"],
            output_dir / f"global_umap_full_{group_name}.png",
            title=f"Global UMAP - Full ({group_name})"
        )

        plot_knn_heatmap(
            group_data["full"],
            group_data["labels"],
            output_dir / f"global_knn_full_{group_name}.png",
            title=f"1-NN Accuracy - Full ({group_name})"
        )





def main():
    parser = argparse.ArgumentParser(
        description="Analyze agreement between predicted and true phyloP scores"
    )
    parser.add_argument(
        "--bigwig_file",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/241-mammalian-2020v2.bigWig",
        help="Path to the bigwig file with phyloP scores",
    )
    parser.add_argument(
        "--genome_fasta",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/hg38.ml.fa",
        help="Path to the genome fasta file",
    )
    parser.add_argument(
        "--gtf_files",
        type=str,
        nargs="+",
        default=['/home/mica/gamba/data_processing/data/240-mammalian/chr2.gtf', '/home/mica/gamba/data_processing/data/240-mammalian/chr19.gtf', '/home/mica/gamba/data_processing/data/240-mammalian/chr22.gtf'],
        help="Path to GTF annotation files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/global_representations",
        help="Directory to save analysis results",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default='/home/mica/gamba/',
        help="Directory containing model checkpoints",
    )
    parser.add_argument(
        "--config_fpath",
        type=str,
        default='/home/mica/gamba/configs/jamba-small-240mammalian.json',
        help="Path to model config JSON",
    )
    parser.add_argument(
        "--num_regions",
        type=int,
        default=1000,
        help="Number of regions to sample per category",
    )
    parser.add_argument(
        "--region_length",
        type=int,
        default=2048,
        help="Length of each sampled region",
    )
    parser.add_argument(
        "--chromosomes",
        type=str,
        nargs="+",
        default=["chr2", "chr19", "chr22"],
        help="List of chromosomes to analyze",
    )
    parser.add_argument(
        "--training_chromosomes",
        type=str,
        nargs="+",
        default=["chr19"],
        help="List of chromosomes used in training",
    )
    parser.add_argument(
        "--test_chromosomes",
        type=str,
        nargs="+",
        default=["chr2", "chr22"],
        help="List of chromosomes held out for testing",
    )
    parser.add_argument(
        "--last_step",
        type=int,
        default=44000,
        help="Checkpoint step to use",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for model predictions",
    )
    parser.add_argument(
        "--model_type", type=str, choices=["gamba", "caduceus"], required=True,
        help="Which model type to use (gamba or caduceus)"
    )
    parser.add_argument(
        "--training_task", type=str, choices=["dual", "cons_only", "seq_only"], required=True,
        help="Which task the model was trained on"
    )

    args = parser.parse_args()
    
    # Configure logging to include timestamps
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logging.info(f"Starting analysis script with chromosomes: {args.chromosomes}")
    logging.info(f"Training chromosomes: {args.training_chromosomes}")
    logging.info(f"Test chromosomes: {args.test_chromosomes}")
    logging.info(f"Using model type: {args.model_type} on task: {args.training_task}")

    if args.model_type == 'gamba':
        checkpoint_dir = args.checkpoint_dir + f"/clean_dcps/CCP/"
    else:
        checkpoint_dir = args.checkpoint_dir + f"/clean_caduceus_dcps/"
    if args.last_step ==0:
        last_step = "random_init"
    else:
        last_step = args.last_step
    #change outputdir to + dcp checkpoint 
    output_dir = args.output_dir + f"/{args.model_type}_{args.training_task}_step_{last_step}/"
    try:
        analyze_agreement(
            args.genome_fasta,
            args.bigwig_file,
            args.gtf_files,
            checkpoint_dir,
            args.config_fpath,
            output_dir,
            num_regions=args.num_regions,
            region_length=args.region_length,
            chromosomes=args.chromosomes,
            training_chromosomes=args.training_chromosomes,
            test_chromosomes=args.test_chromosomes,
            last_step=args.last_step,
            batch_size=args.batch_size,
            training_task= args.training_task,
            model_type=args.model_type
        )
        logging.info("Analysis completed successfully")
    except Exception as e:
        logging.error(f"Error in analysis: {e}")
        import traceback
        logging.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()