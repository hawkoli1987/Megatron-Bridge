
# Call Stacks
```bash
Megatron-LM/megatron/core/
# implements different types of schedulers in phases, 
# core function: OptimizerParamScheduler.get_lr()
- optimizer_param_scheduler.py 

Megatron-Bridge/megatron/bridge/
# use _get_scheduler() to get OptimizerParamScheduler from mcore
# use setup_optimizer() to instantiate the optimizer, after instantiate the scheduler
- training/optim.py 

# define SchedulerConfig, with detail writeup on the args, 
# in ConfigContainer.validate(), convert the duration args from global steps to iters (num_samples)
- training/config.py 

# define distributed_fused_adam_with_wsd() as a group of scheduler configs
- recipes/utils/optimizer_utils.py
# import distributed_fused_adam_with_wsd()
- recipes/qwen/qwen3_4b.py

# in setup(), use setup_optimizer() to instantiate the scheduler and the optimizer
- training/pretrain.py 
```


# Config
```yaml
scheduler:
  # Learning rate (and weight decay) scheduler settings
  _target_: megatron.bridge.training.config.SchedulerConfig
  
  # Ending weight decay value (at end of training)
  end_weight_decay: 0.033  # float
  
  # Number of iterations to decay learning rate over
  lr_decay_iters: 300000  # int
  
  # If using an alternative unit for LR decay, specify number of steps
  lr_decay_steps: null  # int or null
  
  # Learning rate decay schedule style
  lr_decay_style: cosine  # str
  
  # Fraction of training used for LR warmup
  lr_warmup_fraction: null  # float or null
  
  # Initial LR at start of warmup
  lr_warmup_init: 0.0  # float
  
  # Number of iterations to linearly warm up learning rate
  lr_warmup_iters: 500  # int
  
  # If warmup is specified in another unit, use this
  lr_warmup_steps: null  # int or null
  
  # Iterations for a second-phase LR decay
  lr_wsd_decay_iters: null  # int or null
  
  # Decay style for secondary LR annealing phase
  lr_wsd_decay_style: exponential  # str
  
  # Override any optimizer/scheduler states from a loaded checkpoint
  override_opt_param_scheduler: true  # bool
  
  # Starting weight decay value
  start_weight_decay: 0.033  # float
  
  # Use the checkpoint's saved optimizer/scheduler state on resume
  use_checkpoint_opt_param_scheduler: false  # bool
  
  # Steps over which to increase weight decay from start to end values
  wd_incr_steps: null  # int or null
  
  # Weight decay schedule style
  weight_decay_incr_style: constant  # str
  
  # Steps for any weight decay decay
  wsd_decay_steps: null  # int or null
