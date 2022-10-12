
import math

def get_lr_scheduler_function(scheduler_type, num_warmup_steps=None, num_training_steps=None, num_cycles=0.5):
    if scheduler_type != "keep_the_same":
        assert num_warmup_steps is not None
        assert num_training_steps is not None
    
    if scheduler_type == 'linear':
        return linear_schedule_with_warmup_function(num_warmup_steps, num_training_steps)
    elif scheduler_type == 'cosine':
        return cosine_schedule_with_warmup_function(num_warmup_steps, num_training_steps, num_cycles)
    elif scheduler_type == 'cosine_with_hard_restarts':
        return cosine_schedule_with_hard_restarts_function(num_warmup_steps, num_training_steps, num_cycles)
    elif scheduler_type == 'polynomial_decay':
        return polynomial_decay_schedule_with_warmup_function(num_warmup_steps, num_training_steps)
    elif scheduler_type == 'keep_the_same':
        return keep_the_same_function()
    else:
        raise ValueError(f"Unknown scheduler_type: {scheduler_type}")


def linear_schedule_with_warmup_function(num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
        return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            )
    return lr_lambda

def cosine_schedule_with_warmup_function(num_warmup_steps, num_training_steps, num_cycles=0.5):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return lr_lambda

def cosine_schedule_with_hard_restarts_function(num_warmup_steps, num_training_steps, num_cycles=0.5):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        if progress >= 1.0:
            return 0.0
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))

    return lr_lambda

def polynomial_decay_schedule_with_warmup_function(num_warmup_steps, num_training_steps, lr_init=1e-4, lr_end=1e-7, power=1.0):
    if not (lr_init > lr_end):
        raise ValueError(f"lr_end ({lr_end}) must be be smaller than initial lr ({lr_init})")
    
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step > num_training_steps:
            return lr_end / lr_init  # as LambdaLR multiplies by lr_init
        else:
            lr_range = lr_init - lr_end
            decay_steps = num_training_steps - num_warmup_steps
            pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
            decay = lr_range * pct_remaining**power + lr_end
            return decay / lr_init  # as LambdaLR multiplies by lr_init
        
    return lr_lambda

def keep_the_same_function():
    def lr_lambda():
        return 1.0
    return lr_lambda