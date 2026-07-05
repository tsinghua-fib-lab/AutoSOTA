import logging
import os
import socket
import time

from googleapiclient import discovery
from googleapiclient.errors import HttpError


class Toxicity:
    def __init__(self, api_key: str | None = None, qps: int | float = 100, max_trials: int = 10):
        # Set up logger
        self.logger = logging.getLogger(__name__)
        
        try:
            api_key = os.getenv("PERSPECTIVE_AI_API_KEY")
        except:
            raise ValueError("Perspective API key isn't provided and set in the environment variable")
        
        self.api_key = api_key
        self.qps = qps
        self.max_trials = max_trials
        self.rate_limit_enabled = qps != "inf" and qps != float('inf')
        self.min_interval = 1.0 / qps if self.rate_limit_enabled else 0
        self.last_request_time = 0
        
        # Initialize Perspective API client
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize or re-initialize the Perspective API client."""
        self.client = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=self.api_key,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False,
        )
        socket.setdefaulttimeout(30)
    
    def _rate_limit(self):
        if not self.rate_limit_enabled:
            return
        
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_interval:
            sleep_time = self.min_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def predict(self, texts: str | list[str]) -> tuple[list[dict], list[float]]:
        """
        Predict toxicity scores for texts.
        Empty or whitespace-only texts are assigned neutral score (0.0).
        
        Args:
            texts: Single string or list of strings to evaluate
            
        Returns:
            analysis: List of API response dicts (None for empty inputs)
            toxicity: List of toxicity scores (0.0 for empty inputs)
        """
        # Convert single string to list
        if isinstance(texts, str):
            texts = [texts]
        
        analysis, toxicity = [], []
        
        for text in texts:
            # Handle empty or whitespace-only text
            if not text or not text.strip() or len(text.strip()) < 3:
                analysis.append(None)
                toxicity.append(0.0)
                continue
            
            # Retry logic for service unavailability
            for trial in range(self.max_trials):
                try:
                    self._rate_limit()
                    
                    analyze_request = {
                        'comment': {'text': text},
                        'requestedAttributes': {'TOXICITY': {}},
                        'languages': ['en'],
                        'doNotStore': True,
                    }
                    
                    response = self.client.comments().analyze(body=analyze_request).execute()
                    analysis.append(response)
                    toxicity.append(response['attributeScores']['TOXICITY']['summaryScore']['value'])
                    break  # Success, exit retry loop
                    
                except (HttpError, TimeoutError, socket.timeout, ConnectionError, OSError) as e:
                    error_msg = str(e)
                    
                    # Handle empty comment error (backup in case pre-check missed it)
                    if 'COMMENT_EMPTY' in error_msg:
                        analysis.append(None)
                        toxicity.append(0.0)
                        break
                    
                    # Handle other errors with retry and client re-initialization
                    if trial < self.max_trials - 1:
                        # Re-initialize client for connection/network errors
                        if not isinstance(e, HttpError) or (isinstance(e, HttpError) and e.resp.status >= 500):
                            try:
                                self.logger.info("Re-initializing API client due to connection/server error")
                                self._initialize_client()
                            except Exception as init_error:
                                self.logger.warning(f"Failed to re-initialize client: {init_error}")
                        
                        # Exponential backoff: 10s, 30s, 60s for trials 0, 1, 2
                        wait_time = 10 * (2 ** trial) if trial == 0 else 30 * (trial)
                        self.logger.warning(f"Error occurred: {error_msg}. Retrying in {wait_time}s... (attempt {trial + 1}/{self.max_trials})")
                        time.sleep(wait_time)
                    else:
                        # Max trials reached, re-raise the error
                        raise

        return analysis, toxicity
