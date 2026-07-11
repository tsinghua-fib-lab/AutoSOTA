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

from evodiff.utils import Tokenizer
from gamba.constants import TaskType, DNA_ALPHABET_PLUS
from gamba.collators import gLMCollator
from gamba.model import create_model, JambagambaModel, JambaGambaNoConsModel

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


def load_model(checkpoint_dir, config_fpath, last_step=52000, device=None):
    """Load the model from a checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # Get the latest checkpoint path
    ckpt_path = get_latest_dcp_checkpoint_path(checkpoint_dir, last_step=last_step)
    # For reproducibility, use the same checkpoint as in the original code
    ckpt_path = '/home/mica/gamba/clean_dcps/dcp_nocons_56000'
    
    # Load configuration
    with open(config_fpath, "r") as f:
        config = json.load(f)
    
    # Setup tokenizer and task
    tokenizer = Tokenizer(DNA_ALPHABET_PLUS)
    task = TaskType(config["task"].lower().strip())
    
    logging.info(f"Task: {task}, Model: {config['model_type']}, Config: {config['model_config']}")
    
    # Create model
    model, block = create_model(
        task, config["model_type"], config["model_config"], tokenizer.mask_id.item(), 
    )
    
    # Get model parameters from config
    d_model = config.get("d_model", 512)
    nhead = config.get("n_head", 8)
    n_layers = config.get("n_layers", 6)
    dim_feedforward = config.get("dim_feedforward", d_model)
    padding_id = config.get("padding_id", 0)
    
    # Set up the full model
    # model = JambagambaModel(
    #     model, d_model=d_model, nhead=nhead, n_layers=n_layers,
    #     padding_id=padding_id, dim_feedfoward=dim_feedforward
    # )
    model = JambaGambaNoConsModel(
        model, d_model=d_model, nhead=nhead, n_layers=n_layers,
        padding_id=padding_id, dim_feedfoward=dim_feedforward
    )
    
    # Load checkpoint
    logging.info(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(os.path.join(ckpt_path, "model_optimizer.pt"), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Move model to device
    model.to(device)
    model.eval()
    
    return model, tokenizer


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

def sample_regions_by_phylop(bigwig_file, genome_fasta, score_range, num_regions=100, max_length=2048, region_length=None, feature_length=1000, chromosomes=None):
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
                
                # Evaluate only the last feature_length portion
                feature_start_in_window = max(0, len(scores) - feature_length)
                feature_end_in_window = len(scores)
                
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

def sample_regions_by_feature(bigwig_file, genome_fasta, gtf_parser, feature_type, num_regions=100, max_length=2048, chromosomes=None):
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
            
            # Calculate maximum context we can add
            max_context_length = min(max_length - feature_length, 1000)
            
            # If feature is too long, just take the full max_length
            if feature_length > max_length:
                if feature['strand'] == '+':
                    start = feature_start
                    end = feature_start + max_length
                else:
                    # For - strand, prioritize the 3' end of the feature (which is the 5' end of the gene)
                    end = feature_end
                    start = max(1, feature_end - max_length + 1)
                
                feature_start_in_window = 0
                feature_end_in_window = min(feature_length, max_length)
            else:
                # Add context before the feature
                if feature['strand'] == '+':
                    context_start = max(1, feature_start - max_context_length)
                    start = context_start
                    end = feature_end
                else:
                    # For - strand, context should be added after the feature in genomic coordinates
                    # (which is before the feature in gene orientation)
                    context_end = feature_end + max_context_length
                    chrom_length = len(genome[chrom])
                    context_end = min(context_end, chrom_length)
                    start = feature_start
                    end = context_end
                
                # Calculate feature positions within the window
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

def predict_scores_batched(model, tokenizer, regions, batch_size=8, device=None):
    """Run predictions on sampled regions with batching for speed."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create collator
    collator = gLMCollator(tokenizer=tokenizer)
    
    all_predictions = []
    all_true_scores = []
    region_info = []
    
    logging.info(f"Running predictions on {len(regions)} regions with batch size {batch_size}...")
    
    # Process in batches
    for i in tqdm(range(0, len(regions), batch_size), desc="Batch predictions"):
        batch_regions = regions[i:i+batch_size]
        batch_inputs = []
        
        # Prepare batch
        for region in batch_regions:
            # Tokenize sequence
            sequence_tokens = tokenizer.tokenizeMSA(region['sequence'])
            scores = region['scores']
            
            # Store for collation
            batch_inputs.append((sequence_tokens, scores))
            
            # Store region info - use get for mean_score with a sensible default
            region_info.append({
                'chrom': region['chrom'],
                'start': region['start'],
                'end': region['end'],
                'feature_id': region.get('feature_id', 'unknown'),
                'mean_score': region.get('mean_score', 0.0)  # Use default of 0.0 if not present
            })
            
            # Store true scores
            all_true_scores.append(scores)
        
        # Skip empty batches
        if not batch_inputs:
            continue
            
        # Collate batch
        collated = collator(batch_inputs)
        
        # Run the model
        with torch.no_grad():
            output = model(collated[0].to(device), collated[1].to(device))
        
        # Process each item in the batch
        if "scaling_logits" in output:
            scaling_logits = output["scaling_logits"]
        else:
            #set them all to 0
            scaling_logits = torch.zeros((len(batch_inputs), 1, 1), device=device)
        
        # For each item in the batch
        for j in range(scaling_logits.size(0)):
            # Get scaling logits for this sequence
            seq_logits = scaling_logits[j]
            
            # Extract means
            means = seq_logits[:, 0].cpu().numpy()
            
            # Store predictions
            all_predictions.append(means)
    
    # Ensure all arrays have the same length before processing
    min_length = min(len(arr) for arr in all_true_scores + all_predictions)
    
    # Trim arrays to the minimum length
    all_predictions = [arr[:min_length] for arr in all_predictions]
    all_true_scores = [arr[:min_length] for arr in all_true_scores]
    
    return all_predictions, all_true_scores, region_info


def calculate_correlations(true_scores, predicted_scores, region_info, feature_length=1000):
    """
    Calculate correlations between true and predicted scores for each region's feature portion.
    For phyloP-based regions, evaluates only the last feature_length bases.
    For feature-based regions, uses the designated feature portion.
    """
    results = []
    
    for i in range(len(true_scores)):
        # Get the full sequence scores
        true = true_scores[i]
        pred = predicted_scores[i]
        
        # Extract the portion to evaluate
        if 'feature_start_in_window' in region_info[i] and 'feature_end_in_window' in region_info[i]:
            # Use the actual feature boundaries for feature-based regions
            feature_start = region_info[i]['feature_start_in_window']
            feature_end = region_info[i]['feature_end_in_window']
        else:
            # For phyloP-based regions, use the last feature_length bases
            feature_start = max(0, len(true) - feature_length)
            feature_end = len(true)
        
        # Slice to get just the feature/evaluation portion
        true_feature = true[feature_start:feature_end]
        pred_feature = pred[feature_start:feature_end]
        
        # Filter out NaN values
        mask = ~(np.isnan(true_feature) | np.isnan(pred_feature))
        true_filtered = true_feature[mask]
        pred_filtered = pred_feature[mask]
        
        # Calculate correlation if we have enough data points
        if len(true_filtered) > 10:
            correlation = np.corrcoef(true_filtered, pred_filtered)[0, 1]
            mean_true = np.mean(true_filtered)
            mean_pred = np.mean(pred_filtered)
            
            results.append({
                'chrom': region_info[i]['chrom'],
                'start': region_info[i]['start'],
                'end': region_info[i]['end'],
                'feature_id': region_info[i].get('feature_id', 'unknown'),
                'mean_true_score': mean_true,
                'mean_pred_score': mean_pred,
                'correlation': correlation,
                'num_points': len(true_filtered),
                'feature_length': feature_end - feature_start
            })
    
    return pd.DataFrame(results)

def create_feature_comparison_plot(data, output_dir):
    """
    Create bar plot comparing correlations by feature category and data split.
    """
    logging.info("Creating feature comparison bar plot")
    
    # Aggregate data by category and data_split
    agg_data = data.groupby(['category', 'data_split'])['correlation'].agg(
        ['mean', 'std', 'count']
    ).reset_index()
    
    # Pivot for easier plotting
    pivot_data = agg_data.pivot(index='category', columns='data_split', values=['mean', 'std'])
    
    # Sort categories by training performance (if available)
    if ('mean', 'Training') in pivot_data.columns:
        pivot_data = pivot_data.sort_values(by=('mean', 'Training'), ascending=False)
    
    # Extract the necessary data for plotting
    categories = pivot_data.index
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    
    # Set up positions for bars
    x = np.arange(len(categories))
    width = 0.35
    
    # Plot bars for Held Out
    if ('mean', 'Held Out') in pivot_data.columns:
        held_out_means = pivot_data[('mean', 'Held Out')].values
        held_out_stds = pivot_data[('std', 'Held Out')].values if ('std', 'Held Out') in pivot_data.columns else None
        
        bars1 = plt.bar(x - width/2, held_out_means, width, 
                        label='Held Out (chr2, chr22)',
                        color='skyblue', 
                        yerr=held_out_stds,
                        capsize=5,
                        edgecolor='darkblue',
                        linewidth=1.5,
                        alpha=0.8)
        
        # Add correlation values above bars
        for i, v in enumerate(held_out_means):
            plt.text(i - width/2, v + 0.02, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
    
    # Plot bars for Training
    if ('mean', 'Training') in pivot_data.columns:
        training_means = pivot_data[('mean', 'Training')].values
        training_stds = pivot_data[('std', 'Training')].values if ('std', 'Training') in pivot_data.columns else None
        
        bars2 = plt.bar(x + width/2, training_means, width,
                        label='Training (chr19)',
                        color='orange',
                        yerr=training_stds,
                        capsize=5,
                        edgecolor='darkred',
                        linewidth=1.5,
                        alpha=0.8)
        
        # Add correlation values above bars
        for i, v in enumerate(training_means):
            plt.text(i + width/2, v + 0.02, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
    
    # Add horizontal line at y=0 for reference
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    
    # Add statistical significance if available
    try:
        from scipy import stats
        held_out_corrs = data[data['data_split'] == 'Held Out']['correlation']
        training_corrs = data[data['data_split'] == 'Training']['correlation']
        
        if len(held_out_corrs) > 0 and len(training_corrs) > 0:
            t_stat, p_value = stats.ttest_ind(held_out_corrs, training_corrs, equal_var=False)
            
            significance_text = f"T-test: t={t_stat:.2f}, p={p_value:.4f}\n"
            significance_text += f"Mean (Held Out): {held_out_corrs.mean():.3f}\n"
            significance_text += f"Mean (Training): {training_corrs.mean():.3f}"
            
            plt.figtext(0.7, 0.02, significance_text, 
                       bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'),
                       fontsize=10)
    except Exception as e:
        logging.warning(f"Could not add statistical significance: {e}")
    
    # Customize the plot
    plt.xlabel('Feature Category', fontsize=12, fontweight='bold')
    plt.ylabel('Mean Correlation (Predicted vs True PhyloP)', fontsize=12, fontweight='bold')
    plt.title('Correlation Between Predicted and True PhyloP Scores:\nHeld Out vs Training Chromosomes', 
              fontsize=14, fontweight='bold')
    plt.xticks(x, categories, rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(axis='y', alpha=0.3)
    
    # Improve layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_comparison_bar_plot.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'feature_comparison_bar_plot.pdf'))
    plt.close()
    
    logging.info(f"Bar plot saved to {output_dir}")


def create_feature_chromosome_heatmap(data, output_dir):
    """
    Create heatmap showing correlations by feature category and chromosome.
    """
    logging.info("Creating feature/chromosome heatmap")
    
    # Aggregate data by category and chromosome
    agg_data = data.groupby(['category', 'chrom'])['correlation'].mean().reset_index()
    
    # Create pivot table
    pivot_data = agg_data.pivot(index='category', columns='chrom', values='correlation')
    
    # Sort categories (rows) by the average correlation across all chromosomes
    pivot_data['avg'] = pivot_data.mean(axis=1)
    pivot_data = pivot_data.sort_values('avg', ascending=False)
    pivot_data = pivot_data.drop('avg', axis=1)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Define custom colormap (white to blue)
    colors = [(1, 1, 1), (0, 0.4, 0.8)]  # white to blue
    cmap = LinearSegmentedColormap.from_list('white_to_blue', colors, N=100)
    
    # Create heatmap
    heatmap = sns.heatmap(
        pivot_data,
        annot=True,
        fmt='.3f',
        cmap=cmap,
        linewidths=0.5,
        cbar_kws={'label': 'Mean Correlation'},
        annot_kws={"size": 10, "weight": "bold"}
    )
    
    # Add data split information
    for i, chrom in enumerate(pivot_data.columns):
        split = "Training" if chrom == 'chr19' else "Held Out"
        color = "darkred" if chrom == 'chr19' else "navy"
        plt.text(
            i + 0.5, 
            len(pivot_data) + 0.1, 
            split,
            horizontalalignment='center',
            color=color,
            fontsize=10,
            fontweight='bold'
        )
    
    # Customize the plot
    plt.title('Mean Correlation by Feature Category and Chromosome', fontsize=14, fontweight='bold')
    plt.yticks(rotation=0, fontsize=10)
    plt.xticks(fontsize=10, rotation=0)
    
    # Improve layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_chromosome_heatmap.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'feature_chromosome_heatmap.pdf'))
    plt.close()
    
    logging.info(f"Heatmap saved to {output_dir}")


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
    last_step=132250,
    batch_size=8,
    training_chromosomes=None,
    test_chromosomes=None
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
    model, tokenizer = load_model(checkpoint_dir, config_fpath, last_step=last_step, device=device)
    
    # Parse GTF files
    gtf_parser = GTFParser(gtf_files)
    
    # Get phyloP score distribution
    score_dist = get_phylop_score_ranges(bigwig_file, chromosomes, num_samples=1000)
    
    # Define sampling categories
    categories = [
        # Score-based categories
        {"name": "negative_phylop", "type": "score", "range": score_dist['negative']},
        {"name": "neutral_phylop", "type": "score", "range": score_dist['neutral']},
        {"name": "positive_phylop", "type": "score", "range": score_dist['positive']},
        
        # Feature-based categories
        {"name": "coding_regions", "type": "feature", "feature": "coding_regions"},
        {"name": "noncoding_regions", "type": "feature", "feature": "noncoding_regions"},
        {"name": "exons", "type": "feature", "feature": "exons"},
        {"name": "introns", "type": "feature", "feature": "introns"},
        {"name": "promoters", "type": "feature", "feature": "promoters"}
    ]
    
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
                    chromosomes=group_chroms
                )
            else:  # feature-based
                regions = sample_regions_by_feature(
                    bigwig_file, genome_fasta, gtf_parser, category['feature'],
                    num_regions=num_regions, max_length=region_length,
                    chromosomes=group_chroms
                )
            
            if not regions:
                logging.warning(f"No regions sampled for {category['name']} in {group_name} chromosomes")
                continue
            
            # Predict scores for sampled regions
            predicted_scores, true_scores, region_info = predict_scores_batched(
                model, tokenizer, regions, batch_size=batch_size, device=device
            )
            
            # Calculate correlations
            correlation_df = calculate_correlations(true_scores, predicted_scores, region_info)
            
            # Add category information
            correlation_df['category'] = category['name']
            correlation_df['group'] = group_name
            
            # Append to all results
            all_correlation_results.append(correlation_df)
            
            # Log summary stats
            if not correlation_df.empty:
                mean_corr = correlation_df['correlation'].mean()
                logging.info(f"{group_name} - {category['name']}: Mean correlation: {mean_corr:.4f}")
    
    # Combine all results
    all_results_df = pd.concat(all_correlation_results, ignore_index=True)
    
    # Save combined results
    all_results_path = output_dir / "all_correlation_results.csv"
    all_results_df.to_csv(all_results_path, index=False)
    
    # Add data_split column for visualizations
    all_results_df['data_split'] = all_results_df['chrom'].apply(
        lambda x: 'Training' if x in training_chromosomes else 'Held Out'
    )
    
    # Create the two key visualizations
    create_feature_comparison_plot(all_results_df, output_dir)
    create_feature_chromosome_heatmap(all_results_df, output_dir)
    
    # Return the results dataframe for further analysis if needed
    return all_results_df

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
        default="/home/mica/gamba/data_processing/data/240-mammalian/phylop_corr_analysis",
        help="Directory to save analysis results",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default='/home/mica/gamba/dcps/',
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
        default=500,
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
        default=183000,
        help="Checkpoint step to use",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for model predictions",
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
    
    #change outputdir to + dcp checkpoint 
    output_dir = args.output_dir + f"/{args.last_step}/"
    try:
        analyze_agreement(
            args.genome_fasta,
            args.bigwig_file,
            args.gtf_files,
            args.checkpoint_dir,
            args.config_fpath,
            output_dir,
            num_regions=args.num_regions,
            region_length=args.region_length,
            chromosomes=args.chromosomes,
            training_chromosomes=args.training_chromosomes,
            test_chromosomes=args.test_chromosomes,
            last_step=args.last_step,
            batch_size=args.batch_size
        )
        logging.info("Analysis completed successfully")
    except Exception as e:
        logging.error(f"Error in analysis: {e}")
        import traceback
        logging.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()