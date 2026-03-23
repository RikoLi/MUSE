from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    """
    An abstract base class for all trainers.
    """
    logger = None
    def __init__(self, cfg, logger, device, model, optimizer, scheduler, evaluator):
        BaseTrainer.logger = logger
        
        self.cfg = cfg
        self.device = device
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.evaluator = evaluator

    @abstractmethod
    def train(self):
        """Specific training loop should be defined in subclass."""
        raise NotImplementedError
    
    @abstractmethod
    def eval(self):
        """Specific evaluation process should be defined in subclass."""
        raise NotImplementedError
    
class AccelerateBaseTrainer(ABC):
    """An abstract base class for all trainers that use `accelerate`."""
    logger = None
    def __init__(self, cfg, logger, model, optimizer, scheduler, evaluator, accelerator):
        AccelerateBaseTrainer.logger = logger
        self.cfg = cfg
        self.accelerator = accelerator
        self.device = accelerator.device
        self.model, self.optimizer, self.scheduler = model, optimizer, scheduler
        self.evaluator = evaluator
        
    @abstractmethod
    def train(self):
        """Specific training loop should be defined in subclass."""
        raise NotImplementedError
    
    @abstractmethod
    def eval(self):
        """Specific evaluation process should be defined in subclass."""
        raise NotImplementedError