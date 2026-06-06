from .. import arguments

def get_default_arg()->tuple[arguments.ModelParams,arguments.OptimizationParams,arguments.PipelineParams,arguments.DensifyParams]:
    lp=arguments.ModelParams.get_class_default_obj()
    op=arguments.OptimizationParams.get_class_default_obj()
    pp=arguments.PipelineParams.get_class_default_obj()
    dp=arguments.DensifyParams.get_class_default_obj()
    return lp,op,pp,dp

def get_quality_arg()->tuple[arguments.OptimizationParams,arguments.PipelineParams,arguments.DensifyParams]:
    lp=arguments.ModelParams.get_class_default_obj()
    op=arguments.OptimizationParams.get_class_default_obj()
    pp=arguments.PipelineParams.get_class_default_obj()
    dp=arguments.DensifyParams.get_class_default_obj()
    dp.densify_grad_threshold=0.00015
    return lp,op,pp,dp