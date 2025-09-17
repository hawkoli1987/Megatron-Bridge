
# Data Pipeline Concepts and Call Stack

## 1. Dataset Configuration

**Definition:** A configuration class (dataclass) that contains all parameters needed to build a specific type of dataset.

**Purpose:**
- Stores dataset-specific parameters (paths, tokenization settings, etc.)
- Provides type safety and validation
- Enables configuration-driven dataset creation
- Separates configuration from implementation

### GPT Dataset (Pretraining)
**File:** `src/megatron/bridge/training/config.py`
```python
@dataclass
class GPTDatasetConfig(MCoreGPTDatasetConfig, DataloaderConfig):  # Lines 174-188
    skip_getting_attention_mask_from_dataset: bool = True
    reset_position_ids: Optional[bool] = None
    reset_attention_mask: Optional[bool] = None
    eod_mask_loss: Optional[bool] = None
```

## 2. Dataset Class Implementations

**Definition:** A Dataset is a Python class that implements the PyTorch `Dataset` interface and provides access to individual data samples.

**Purpose:**
- Loads and preprocesses raw data from files
- Implements `__getitem__()` to return individual samples
- Implements `__len__()` to return dataset size
- Handles tokenization, padding, truncation, and other data transformations

### GPT Dataset (Pretraining)
**File:** `src/megatron/bridge/data/datasets/sft.py`
```python
class GPTSFTDataset(Dataset):  # Lines 183-258
    def __init__(self, file_path: str, tokenizer: MegatronTokenizer, ...):  # Lines 186-217
        """Loads JSONL data for supervised fine-tuning"""
        
    def __getitem__(self, idx: int) -> dict:
        """Returns a single training example:
        {
            'input_ids': torch.tensor([1, 2, 3, 4, 5]),
            'labels': torch.tensor([1, 2, 3, 4, 5]), 
            'loss_mask': torch.tensor([0, 0, 0, 1, 1]),
            'attention_mask': torch.tensor([1, 1, 1, 1, 1]),
            'position_ids': torch.tensor([0, 1, 2, 3, 4])
        }
        """
        
    def __len__(self) -> int:
        """Returns number of samples in dataset"""
```

### Dataset Builder Classes
**File:** `src/megatron/bridge/data/builders/finetuning_dataset.py`
```python
class FinetuningDatasetBuilder:
    def build(self) -> list[Optional[Any]]:
        """Builds train, validation, and test datasets"""
```

**File:** `src/megatron/bridge/data/builders/hf_dataset.py`
```python
class HFDatasetBuilder(FinetuningDatasetBuilder):
    def __init__(self, dataset_name: str, tokenizer, ...):
        """Builds HuggingFace datasets"""
```

## 3. Dataset Providers

**Definition:** A factory function that takes configuration parameters and returns dataset instances (train, validation, test).

**Purpose:**
- Abstracts dataset creation logic
- Maps configuration types to appropriate dataset builders
- Provides a unified interface for different dataset types
- Handles the complexity of dataset instantiation

**File:** `src/megatron/bridge/data/utils.py`

### GPT Pretraining Dataset Provider
```python
def pretrain_train_valid_test_datasets_provider(  # Lines 49-51
    train_val_test_num_samples: list[int], dataset_config: BlendedMegatronDatasetConfig
) -> tuple[GPTDataset, GPTDataset, GPTDataset]:
    """Creates GPT datasets for pretraining"""
    
    if dataset_config.mock:
        dataset_type = MockGPTDataset
    else:
        dataset_type = GPTDataset

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type, train_val_test_num_samples, lambda: True, dataset_config
    ).build()

    return train_ds, valid_ds, test_ds
```

### HuggingFace Dataset Provider
...

### Finetuning Dataset Provider
...

### Provider Registry
```python
_REGISTRY: Dict[Type[Union[FinetuningDatasetConfig, BlendedMegatronDatasetConfig, HFDatasetConfig]], Callable] = {
    GPTDatasetConfig: pretrain_train_valid_test_datasets_provider,
    MockGPTDatasetConfig: pretrain_train_valid_test_datasets_provider,
    HFDatasetConfig: hf_train_valid_test_datasets_provider,
    FinetuningDatasetConfig: finetuning_train_valid_test_datasets_provider,
}

def get_dataset_provider(dataset_config) -> Callable:
    """Returns the appropriate provider function based on config type"""
    return _REGISTRY[type(dataset_config)]
```

## 4. DataLoader

**Definition:** A PyTorch `DataLoader` that wraps a Dataset and provides batching, shuffling, and multi-process data loading.

**Purpose:**
- Batches individual samples from Dataset
- Handles data loading parallelism with multiple workers
- Manages memory pinning and prefetching
- Provides data shuffling and sampling strategies

### Dataset to DataLoader Conversion
**File:** `src/megatron/bridge/data/loaders.py`
```python
def build_train_valid_test_data_loaders(  # Lines 165-167
    cfg: ConfigContainer, 
    train_state: TrainState, 
    build_train_valid_test_datasets_provider: Callable
) -> tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    """Creates PyTorch DataLoaders from datasets"""
    
    # First build the datasets
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        cfg=cfg, 
        build_train_valid_test_datasets_provider=build_train_valid_test_datasets_provider
    )
    
    # Create DataLoaders with specific configurations
    train_dataloader = build_pretraining_data_loader(
        train_ds,                           # Dataset instance
        train_state.consumed_train_samples, # Starting sample index
        cfg.dataset.dataloader_type,        # "single", "cyclic", "external"
        cfg.train.micro_batch_size,         # Batch size (e.g., 4)
        cfg.dataset.num_workers,            # Number of worker processes (e.g., 4)
        cfg.dataset.data_sharding,          # Whether to shard data across ranks
        worker_init_fn=maybe_worker_init_fn, # Worker initialization
        collate_fn=train_ds.collate_fn,     # Custom collation function
        pin_memory=cfg.dataset.pin_memory,  # Memory pinning for GPU transfer
        persistent_workers=cfg.dataset.persistent_workers, # Keep workers alive
        data_parallel_rank=mpu.get_data_parallel_rank(),   # Current rank
        data_parallel_size=mpu.get_data_parallel_world_size(), # Total ranks
    )
```

### DataLoader Output Example
A DataLoader yields batches like:
```python
# Example batch from train_dataloader
batch = {
    "tokens": torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]),      # Shape: [batch_size, seq_length]
    "labels": torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]),      # Shape: [batch_size, seq_length]
    "loss_mask": torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),    # Shape: [batch_size, seq_length]
    "attention_mask": torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]), # Shape: [batch_size, seq_length]
    "position_ids": torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]),  # Shape: [batch_size, seq_length]
}
```

## 5. DataIterator

**Definition:** A wrapper around DataLoader that provides additional functionality like fault tolerance, rerun capabilities, and different iteration strategies.

**Purpose:**
- Wraps PyTorch DataLoader with Megatron-specific features
- Provides fault tolerance and checkpointing support
- Handles different iteration modes (single, cyclic, external)
- Enables data iteration to be restarted from checkpoints

### DataLoader to DataIterator Conversion
**File:** `src/megatron/bridge/data/loaders.py`
```python
def build_train_valid_test_data_iterators(  # Lines 273-275
    cfg: ConfigContainer, 
    train_state: TrainState, 
    build_train_valid_test_datasets_provider: Callable
) -> tuple[Optional[RerunDataIterator], Optional[RerunDataIterator], Optional[RerunDataIterator]]:
    """Creates RerunDataIterators from DataLoaders"""
    
    # Build DataLoaders first
    train_dataloader, valid_dataloader, test_dataloader = build_train_valid_test_data_loaders(...)
    
    # Create iterators based on dataloader type
    dl_type = cfg.dataset.dataloader_type  # "single", "cyclic", "external"
    
    def _get_iterator(dataloader_type, dataloader):
        if dataloader_type == "single":
            return RerunDataIterator(iter(dataloader))           # Single pass through data
        elif dataloader_type == "cyclic":
            return RerunDataIterator(iter(cyclic_iter(dataloader))) # Infinite cycling through data
        elif dataloader_type == "external":
            if isinstance(dataloader, list):
                return [RerunDataIterator(d) for d in dataloader] # Multiple iterators
            else:
                return RerunDataIterator(dataloader)             # External iterator
```

### DataIterator Usage
**File:** `src/megatron/bridge/training/eval.py`
```python
def evaluate(
    state: GlobalState,
    forward_step_func: Callable,
    data_iterator: Optional[Union[RerunDataIterator, list[RerunDataIterator]]],  # DataIterator input
    model: list[MegatronModule],
    ...
) -> tuple[Optional[dict[str, torch.Tensor]], Optional[Any], bool]:
    """Uses DataIterator in evaluation loop"""
    
    while iteration < state.cfg.train.eval_iters:
        # Get next batch from DataIterator
        loss_dicts = forward_backward_func(
            forward_step_func=wrapped_forward_step,
            data_iterator=data_iterator,  # Passed to forward_step_func
            model=model,
            ...
        )
```

## Complete Call Stack

### 1. Entry Point: pretrain() function
**File:** `src/megatron/bridge/training/pretrain.py`
```python
def pretrain(config: ConfigContainer, forward_step_func: Callable) -> None:
    # Step 1: Get dataset provider
    dataset_provider = get_dataset_provider(config.dataset)
    
    # Step 2: Setup everything including data iterators
    setup_output = setup(config, dataset_provider)
    
    # Step 3: Extract data iterators
    valid_data_iterator = setup_output.valid_data_iterator
```

### 2. Dataset Provider Creation
**File:** `src/megatron/bridge/data/utils.py`
```python
def get_dataset_provider(dataset_config) -> Callable:
    return _REGISTRY[type(dataset_config)]
```

### 3. Setup Function Call Stack
**File:** `src/megatron/bridge/training/setup.py`
- setup() calls setup_data_iterators()

**File:** `src/megatron/bridge/data/loaders.py`
- setup_data_iterators() -> build_train_valid_test_data_iterators()
- build_train_valid_test_data_iterators() -> build_train_valid_test_data_loaders()
- build_train_valid_test_data_loaders() -> build_train_valid_test_datasets()
- build_train_valid_test_datasets() -> the dataset provider

## Data Flow Summary

```
Dataset Config → Dataset Provider → Dataset → DataLoader → DataIterator
     ↓              ↓                ↓         ↓           ↓
  Parameters    Factory Function   Raw Data  Batched Data  Fault-Tolerant
  (YAML/JSON)   (Builder Pattern)  (Samples) (Batches)     Iteration
```

### Key Differences:
- **Dataset:** Handles individual samples, tokenization, preprocessing
- **DatasetConfig:** Configuration specification for dataset parameters
- **DatasetProvider:** Factory pattern for creating datasets from configs
- **DataLoader:** Handles batching, multi-processing, memory management
- **DataIterator:** Handles fault tolerance, checkpointing, iteration strategies



```yaml
dataset:
  # Configuration specific to GPT datasets (data loading and processing)
  _target_: megatron.bridge.training.config.GPTDatasetConfig
  
  # Append an extra token to each sequence to ensure both input and output tokens reach the desired sequence length
  add_extra_token_to_sequence: true  # bool
  
  # Blend multiple datasets with specified weights
  blend: null  # tuple or null, format: ([dataset_prefixes], [dataset_weights]) e.g. [["data1","data2"], [0.3,0.7]]
  
  # Per-split dataset blend configurations
  blend_per_split: null  # list or null
  
  # Generate attention masks in the dataset
  create_attention_mask: true  # bool
  
  # Shard the dataset across data-parallel ranks
  data_sharding: true  # bool
  
  # Data loader strategy
  dataloader_type: single  # str, options: 'single', 'cyclic', 'LDDL'
  
  # Drop the last validation batch/sequence if it is smaller than full length
  drop_last_partial_validation_sequence: true  # bool
  
  # Mask loss for end-of-document tokens
  eod_mask_loss: false  # bool
  
  # If true, run validation on the entire validation set (overriding eval_iters)
  full_validation: null  # bool or null
  
  # Fraction of surplus samples to allocate when splitting datasets to avoid running out
  mid_level_dataset_surplus: 0.005  # float
  
  # Memory-map the dataset binary files for efficient reading
  mmap_bin_files: true  # bool
  
  # Use a synthetic/mock dataset (random data) instead of real data
  mock: true  # bool
  
  # Multiple validation sets if any
  multiple_validation_sets: null  # dict or null, e.g. {"val1": path1, "val2": path2}
  
  # Number of threads to use for building the dataset index
  num_dataset_builder_threads: 1  # int
  
  # Number of DataLoader worker processes per training rank
  num_workers: 8  # int
  
  # Local cache directory for index files when loading dataset from object storage
  object_storage_cache_path: null  # str or null
  
  # Path to cache preprocessed dataset shards locally
  path_to_cache: null  # str or null
  
  # Keep DataLoader workers alive between epochs
  persistent_workers: false  # bool
  
  # Pin DataLoader memory (page-locked) for faster host-to-device transfers
  pin_memory: true  # bool
  
  # Random seed for data shuffling and sampling
  random_seed: 1234  # int
  
  # Ignore any attention mask stored in the dataset and recompute it
  reset_attention_mask: false  # bool
  
  # Reset position indices in packed sequences
  reset_position_ids: false  # bool
  
  # Sequence length (number of tokens) per sample for training
  sequence_length: 4096  # int
  
  # Do not fetch precomputed attention masks from dataset
  skip_getting_attention_mask_from_dataset: true  # bool
  
  # Ratios for splitting dataset into train/val/test
  split: 1,1,1  # str
  
  # Cumulative split boundaries corresponding to the 'split' ratios
  split_matrix:
  - - 0
    - 0.3333333333333333
  - - 0.3333333333333333
    - 0.6666666666666666
  - - 0.6666666666666666
    - 1.0  # list of lists
  
  # Tokenizer instance for dataset processing
  tokenizer: null  # object or null
