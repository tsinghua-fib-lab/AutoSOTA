from openood.pipelines.streaming_ood_pipeline import StreamingOODPipeline


def get_pipeline(cfg):
    """
    Factory function: return the pipeline instance for the given config.

    Currently only StreamingOODPipeline is implemented. Add further
    elif branches here when introducing new methods or datasets.
    """
    return StreamingOODPipeline(cfg)
