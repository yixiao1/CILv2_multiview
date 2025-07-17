import wandb
from typing import Optional, Dict, Any
from configs import g_conf

class WandbLogger:
    def __init__(self, project_name: str, experiment_name: str, config: Dict[str, Any], 
                 save_dir: Optional[str] = None):
        self.run = wandb.init(
            project=project_name,
            name=experiment_name,
            config=config,
            dir=save_dir
        )
        
    def log_scalar(self, tag: str, value: float, step: int):
        wandb.log({tag: value}, step=step)
    
    def log_histogram(self, tag: str, values: np.ndarray, step: int):
        wandb.log({tag: wandb.Histogram(values)}, step=step)
    
    def log_image(self, tag: str, images: np.ndarray, step: int):
        wandb.log({tag: [wandb.Image(img) for img in images]}, step=step)
    
    def finish(self):
        wandb.finish()

# Global logger instance
_wandb_logger = None

def create_log(save_full_path: str, train_log_frequency: int = 1, 
               train_image_log_frequency: int = 15) -> None:
    global _wandb_logger
    global TRAIN_LOG_FREQUENCY
    global TRAIN_IMAGE_LOG_FREQUENCY
    
    TRAIN_LOG_FREQUENCY = train_log_frequency
    TRAIN_IMAGE_LOG_FREQUENCY = train_image_log_frequency
    
    # Initialize W&B logger
    _wandb_logger = WandbLogger(
        project_name=g_conf.EXPERIMENT_BATCH_NAME,
        experiment_name=g_conf.EXPERIMENT_NAME,
        config=g_conf.__dict__,
        save_dir=save_full_path
    )

def add_scalar(tag: str, value: float, iteration: int = None) -> None:
    if iteration is not None and iteration % TRAIN_LOG_FREQUENCY == 0:
        _wandb_logger.log_scalar(tag, value, iteration)
        
def add_histogram(tag: str, values: np.ndarray, iteration: int = None) -> None:
    if iteration is not None and iteration % TRAIN_LOG_FREQUENCY == 0:
        _wandb_logger.log_histogram(tag, values, iteration)
        
def add_image(tag: str, images: np.ndarray, iteration: int = None) -> None:
    if iteration is not None and iteration % TRAIN_IMAGE_LOG_FREQUENCY == 0:
        _wandb_logger.log_image(tag, images, iteration)